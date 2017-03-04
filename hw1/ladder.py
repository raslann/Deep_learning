
import torch as T
from torch.autograd import Variable
import torch.nn as NN
import torch.nn.functional as F
from torch.nn import Parameter
import torch.optim as OPT
import numpy as NP
import six
import argparse
import pickle
import json

from modules import BufferList, GlobalAvgPool2d, GlobalUpsample2d


# TODO: make an argparse Parser to parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--unlabeled",
                    help="use unlabeled data",
                    action="store_true")
parser.add_argument("--pseudolabels",
                    help="use pseudo-labels",
                    action="store_true")
parser.add_argument("--modelname",
                    help="model name",
                    type=str)
parser.add_argument("--loadmodel",
                    help="file to load model from for continuation",
                    type=str)

args = parser.parse_args()

from util import alloc_list, anynan, noise, noised
from dataset import fetch, valid_loader

_eps = 1e-8


class Denoiser(NN.Module):
    def __init__(self, size):
        super(Denoiser, self).__init__()

        self.a1 = Parameter(T.zeros(size))
        self.a2 = Parameter(T.ones(size))
        self.a3 = Parameter(T.zeros(size))
        self.a4 = Parameter(T.zeros(size))
        self.a5 = Parameter(T.zeros(size))
        self.a6 = Parameter(T.zeros(size))
        self.a7 = Parameter(T.ones(size))
        self.a8 = Parameter(T.zeros(size))
        self.a9 = Parameter(T.zeros(size))
        self.a10 = Parameter(T.zeros(size))

    def forward(self, z, u):
        assert z.size() == u.size()

        a = [None] * 11
        for i in range(1, 11):
            a[i] = getattr(self, "a%d" % i).unsqueeze(0).expand_as(z)

        mu = a[1] * F.sigmoid(a[2] * u + a[3]) + a[4] * u + a[5]
        nu = a[6] * F.sigmoid(a[7] * u + a[8]) + a[9] * u + a[10]

        return (z - mu) * nu + mu


class Ladder(NN.Module):
    def _add_W(self, input_config, output_config):
        l = NN.Linear(input_config, output_config)
        self.W.append(l)

    def _add_conv(self, inchan, outchan, size):
        l = NN.Conv2d(inchan, outchan, size, padding=(size - 1) / 2)
        self.W.append(l)

    def _add_pool(self, size):
        l = NN.MaxPool2d(size)
        self.W.append(l)

    def _add_gapool(self):
        l = GlobalAvgPool2d()
        self.W.append(l)

    def _add_act(self, act_module):
        self.act.append(act_module)

    def _add_V(self, input_config, output_config):
        l = NN.Linear(input_config, output_config)
        self.V.append(l)

    def _add_deconv(self, inchan, outchan, size):
        l = NN.ConvTranspose2d(inchan, outchan, size, padding=(size - 1) / 2)
        self.V.append(l)

    def _add_unpool(self, size):
        l = NN.UpsamplingNearest2d(scale_factor=size)
        self.V.append(l)

    def _add_gaunpool(self, size):
        l = GlobalUpsample2d(size)
        self.V.append(l)

    def _add_g(self, state_config):
        d = Denoiser(state_config)
        self.g.append(d)

    def _add_stats(self, state_config):
        gamma = Parameter(T.ones(state_config), requires_grad=True)
        beta = Parameter(T.zeros(state_config), requires_grad=True)
        mean = Variable(T.zeros(state_config))
        var = Variable(T.zeros(state_config))
        mean_noisy = Variable(T.zeros(state_config))
        var_noisy = Variable(T.zeros(state_config))
        mean_dec = Variable(T.zeros(state_config))
        var_dec = Variable(T.zeros(state_config))

        self.gamma.append(gamma)
        self.beta.append(beta)
        self.mean.append(mean)
        self.var.append(var)
        self.mean_noisy.append(mean_noisy)
        self.var_noisy.append(var_noisy)
        self.mean_dec.append(mean_dec)
        self.var_dec.append(var_dec)

    def _add_input_stats(self, input_config):
        self.mean_noisy.append(Variable(T.zeros(input_config)))
        self.var_noisy.append(Variable(T.zeros(input_config)))
        self.mean.append(Variable(T.zeros(input_config)))
        self.var.append(Variable(T.zeros(input_config)))
        self.mean_dec.append(Variable(T.zeros(input_config)))
        self.var_dec.append(Variable(T.zeros(input_config)))

    def __init__(self, config, lambda_):
        '''
        Parameters
        ----------
        config: either a file or a string containing network configuration in
                JSON.
        '''
        super(Ladder, self).__init__()

        if isinstance(config, file):
            config = json.load(config)
        elif isinstance(config, str):
            config = json.loads(config)

        layers = config['layers']
        input_size = config['input-size']
        output_size = config['output-size']

        self.L = 0
        self.lambda_ = lambda_

        self.W = NN.ModuleList([None])
        self.act = NN.ModuleList([None])
        self.V = NN.ModuleList([None])
        self.g = NN.ModuleList([])
        self.gamma = NN.ParameterList([None])
        self.beta = NN.ParameterList([None])
        self.mean_noisy = BufferList([])
        self.var_noisy = BufferList([])
        self.mean = BufferList([])
        self.var = BufferList([])
        self.mean_dec = BufferList([])
        self.var_dec = BufferList([])
        self.count = 0

        # Input
        self._add_g((1, input_size, input_size))    # g_0
        self._add_input_stats((1, input_size, input_size))

        size = input_size
        channels = 1

        # Hidden layers
        for layer_conf in layers:
            if layer_conf['type'] == 'conv':
                self._add_conv(layer_conf['inchan'], layer_conf['outchan'], layer_conf['size'])
                self._add_deconv(layer_conf['outchan'], layer_conf['inchan'], layer_conf['size'])
                channels = layer_conf['outchan']
            elif layer_conf['type'] == 'pool':
                self._add_pool(layer_conf['size'])
                self._add_unpool(layer_conf['size'])
                size /= layer_conf['size']
            elif layer_conf['type'] == 'global-pool':
                self._add_gapool()
                self._add_gaunpool(size)
                size = channels
                channels = 0
            elif layer_conf['type'] == 'dense':
                self._add_W(layer_conf['in'], layer_conf['out'])
                self._add_V(layer_conf['out'], layer_conf['in'])
                size = layer_conf['out']

            self._add_act(NN.ReLU())
            if channels != 0:
                # Still in convolution part...
                self._add_g((channels, size, size))
                self._add_stats((channels, size, size))
            else:
                # Now we are in fully-connected part...
                self._add_g(size)
                self._add_stats(size)

            self.L += 1

        # Classifier (softmax)
        self._add_W(size, output_size)
        self._add_act(NN.LogSoftmax())
        self._add_V(output_size, size)
        self._add_g(output_size)
        self._add_stats(output_size)
        self.L += 1

    def running_average(self, avg, new):
        avg.data = (avg.data * (self.count - 1) + new.data) / self.count

    def batchnorm(self, x, l, path):
        '''
        Normalize @x by batch while updating the stats in layer @l.
        '''
        batch_size = x.size()[0]
        if self.training:
            mean = x.mean(0)
            var = x.var(0) * (batch_size - 1) / batch_size
            if path == 'clean':
                self.running_average(self.mean[l], mean)
                self.running_average(self.var[l], var)
            elif path == 'noisy':
                self.running_average(self.mean_noisy[l], mean)
                self.running_average(self.var_noisy[l], var)
            elif path == 'dec':
                self.running_average(self.mean_dec[l], mean)
                self.running_average(self.var_dec[l], var)
            else:
                assert False
        else:
            if path == 'clean':
                mean = self.mean[l].unsqueeze(0)
                var = self.var[l].unsqueeze(0)
            elif path == 'noisy':
                mean = self.mean_noisy[l].unsqueeze(0)
                var = self.var_noisy[l].unsqueeze(0)
            elif path == 'dec':
                mean = self.mean_dec[l].unsqueeze(0)
                var = self.var_dec[l].unsqueeze(0)
            else:
                assert False

        std = (var + _eps).sqrt()
        assert (std.data == 0).sum() == 0
        return (x - mean.expand_as(x)) / std.expand_as(x)

    def forward(self, x):
        self.count += 1
        batch_size = x.size()[0]

        h_tilde = alloc_list(self.L + 1)
        z_tilde = alloc_list(self.L + 1)
        h = alloc_list(self.L + 1)
        z = alloc_list(self.L + 1)
        z_pre = alloc_list(self.L + 1)
        mu = alloc_list(self.L + 1)
        sigma = alloc_list(self.L + 1)
        u = alloc_list(self.L + 1)
        z_hat = alloc_list(self.L + 1)
        z_hat_bn = alloc_list(self.L + 1)

        h_tilde[0] = z_tilde[0] = noised(x)

        for l in range(1, self.L + 1):
            z_tilde[l] = noised(self.batchnorm(self.W[l](h_tilde[l - 1]), l, 'noisy'))
            _beta = self.beta[l].unsqueeze(0).expand_as(z_tilde[l])
            _gamma = self.gamma[l].unsqueeze(0).expand_as(z_tilde[l])
            h_tilde[l] = self.act[l](_gamma * (z_tilde[l] + _beta))

        y_tilde = h_tilde[self.L]

        h[0] = z[0] = x

        for l in range(1, self.L + 1):
            z_pre[l] = self.W[l](h[l - 1])
            if self.training:
                mu[l] = z_pre[l].mean(0)
                sigma[l] = (z_pre[l].var(0) * (batch_size - 1) / batch_size + _eps).sqrt()
            else:
                mu[l] = self.mean[l].unsqueeze(0)
                sigma[l] = (self.var[l] * (batch_size - 1) / batch_size + _eps).sqrt().unsqueeze(0)
            z[l] = self.batchnorm(self.W[l](h[l - 1]), l, 'clean')
            _beta = self.beta[l].unsqueeze(0).expand_as(z_tilde[l])
            _gamma = self.gamma[l].unsqueeze(0).expand_as(z_tilde[l])
            h[l] = self.act[l](_gamma * (z[l] + _beta))

        y = h[self.L]

        for l in range(self.L, -1, -1):
            if l == self.L:
                u[l] = self.batchnorm(h_tilde[self.L], l, 'dec')
            else:
                u[l] = self.batchnorm(self.V[l + 1](z_hat[l + 1]), l, 'dec')
            z_hat[l] = self.g[l](z_tilde[l], u[l])
            if l != 0:
                # Seems that they are not normalizing z_hat on the
                # first layer...
                _mu = mu[l].expand_as(z_hat[l])
                _sigma = sigma[l].expand_as(z_hat[l])
                assert (_sigma.data == 0).sum() == 0
                z_hat_bn[l] = (z_hat[l] - _mu) / _sigma
            else:
                z_hat_bn[l] = z_hat[l]

        rec_loss = 0
        for l in range(0, self.L + 1):
            rec_loss += self.lambda_[l] * ((z[l] - z_hat_bn[l]) ** 2).mean()

        return y_tilde, y, rec_loss

    def reset_stats(self):
        self.count = 0


with open('ladder.json') as config_file:
    model = Ladder(config_file,
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1])
opt = OPT.Adam(model.parameters(), lr=1e-3)
best_acc = 0


def train_model():
    global best_acc
    for E in range(0, 500):
        model.train()
        model.reset_stats()
        for B in range(0, 600 if args.unlabeled else 30):
            (data_l, target_l), (data_u, target_u) = fetch()
            if args.unlabeled:
                data = T.cat([data_l, data_u])
                target = T.cat([target_l, target_u])
            else:
                data = data_l
                target = target_l
            data = (Variable(data).float() / 255.).view(-1, 1, 28, 28)
            target = Variable(target)
            opt.zero_grad()
            y_tilde, _, rec_loss = model(data)

            labeled_mask = (target != -1).unsqueeze(1)
            y_tilde_labeled = y_tilde * labeled_mask.float().expand_as(y_tilde)
            target = target * labeled_mask.long()
            labeled_loss = F.nll_loss(y_tilde_labeled, target)

            if args.unlabeled and args.pseudolabels:
                unlabeled_mask = (target == -1).unsqueeze(1)
                y_tilde_unlabeled = y_tilde * unlabeled_mask.float().expand_as(y_tilde)
                pseudo_labeled_loss = -y_tilde_unlabeled.max(1)[0]
                pseudo_labeled_loss = ((pseudo_labeled_loss * unlabeled_mask.float())* \
                                      NP.min((.5, 1e-1*(E+.1)))).mean()
            else:
                pseudo_labeled_loss = 0
            loss = labeled_loss + rec_loss + pseudo_labeled_loss
            assert not anynan(loss.data)

            loss.backward()
            for p in model.parameters():
                assert not anynan(p.grad.data)
            opt.step()
            six.print_('#%05d      %.5f' % (B, loss.data[0]))

        model.eval()
        valid_loss = 0
        acc = 0
        for B, (data, target) in enumerate(valid_loader):
            data = (Variable(data, volatile=True).float() / 255.).view(-1, 1, 28, 28)
            target = Variable(target, volatile=True)
            y_tilde, y, rec_loss = model(data)
            loss = F.nll_loss(y, target)
            valid_loss += loss.data[0] * data.size()[0]
            acc += (y.data.numpy().argmax(axis=1) == target.data.numpy()).sum()
        valid_loss /= len(valid_loader.dataset)
        six.print_('@%05d      %.5f (%d)' % (E, valid_loss, acc))

        if best_acc < acc:
            best_acc = acc
            if args.modelname is not None:
                with open('%s%d.p' % (args.modelname, E), 'wb') as f:
                    T.save(model, f)

if args.loadmodel is not None:
    with open(args.loadmodel, 'rb') as f:
        model = T.load(f)
train_model()
