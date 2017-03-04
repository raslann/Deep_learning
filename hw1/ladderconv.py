
import torch as T
from torch.autograd import Variable
import torch.nn as NN
import torch.nn.functional as F
from torch.nn import Parameter
import torch.optim as OPT
import numpy as NP
import numpy.random as RNG
import six
import pickle
import json

from modules import BufferList, GlobalAvgPool2d, GlobalUpsample2d, Ensemble

from util import alloc_list, anynan, noise, noised, args, variable, var_to_numpy
from util import batchnorm_mean_var
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
        mean = T.zeros(state_config)
        var = T.zeros(state_config)
        mean_noisy = T.zeros(state_config)
        var_noisy = T.zeros(state_config)
        mean_dec = T.zeros(state_config)
        var_dec = T.zeros(state_config)

        self.gamma.append(gamma)
        self.beta.append(beta)
        self.mean.append(mean)
        self.var.append(var)
        self.mean_noisy.append(mean_noisy)
        self.var_noisy.append(var_noisy)
        self.mean_dec.append(mean_dec)
        self.var_dec.append(var_dec)

    def _add_input_stats(self, input_config):
        self.mean_noisy.append(T.zeros(input_config))
        self.var_noisy.append(T.zeros(input_config))
        self.mean.append(T.zeros(input_config))
        self.var.append(T.zeros(input_config))
        self.mean_dec.append(T.zeros(input_config))
        self.var_dec.append(T.zeros(input_config))

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
        self._add_input_stats(1)

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
                self._add_stats(channels)
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
        avg *= 0.5
        avg += new * 0.5

    def batchnorm(self, x, l, path):
        '''
        Normalize @x by batch while updating the stats in layer @l.
        '''
        if self.training:
            mean, var = batchnorm_mean_var(x)
            #print 'mean', l, path, var_to_numpy(mean).flatten()
            #print 'var', l, path, var_to_numpy(var).flatten()

            if path == 'clean':
                self.running_average(self.mean[l], mean.data)
                self.running_average(self.var[l], var.data)
            elif path == 'noisy':
                self.running_average(self.mean_noisy[l], mean.data)
                self.running_average(self.var_noisy[l], var.data)
            elif path == 'dec':
                self.running_average(self.mean_dec[l], mean.data)
                self.running_average(self.var_dec[l], var.data)
            else:
                assert False
        else:
            if path == 'clean':
                mean = self.mean[l]
                var = self.var[l]
            elif path == 'noisy':
                mean = self.mean_noisy[l]
                var = self.var_noisy[l]
            elif path == 'dec':
                mean = self.mean_dec[l]
                var = self.var_dec[l]
            else:
                assert False

            #print 'mean', l, path, var_to_numpy(mean).flatten()
            #print 'var', l, path, var_to_numpy(var).flatten()

            mean = variable(mean).unsqueeze(0)
            var = variable(var).unsqueeze(0)
            if x.dim() == 4:
                mean = mean.unsqueeze(2).unsqueeze(3)
                var = var.unsqueeze(2).unsqueeze(3)

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

            _beta = self.beta[l].unsqueeze(0)
            _gamma = self.gamma[l].unsqueeze(0)
            if z_tilde[l].dim() == 4:
                _beta = _beta.unsqueeze(2).unsqueeze(3)
                _gamma = _gamma.unsqueeze(2).unsqueeze(3)
            _beta = _beta.expand_as(z_tilde[l])
            _gamma = _gamma.expand_as(z_tilde[l])

            h_tilde[l] = self.act[l](_gamma * (z_tilde[l] + _beta))
            #print 'h_tilde', l, var_to_numpy(h_tilde[l]).max(), var_to_numpy(h_tilde[l]).min()
            assert not anynan(h_tilde[l].data)

        y_tilde = h_tilde[self.L]

        h[0] = z[0] = x

        for l in range(1, self.L + 1):
            z_pre[l] = self.W[l](h[l - 1])
            if self.training:
                mu[l], var = batchnorm_mean_var(z_pre[l])
                sigma[l] = (var + _eps).sqrt()
            else:
                mu[l] = variable(self.mean[l]).unsqueeze(0)
                sigma[l] = (variable(self.var[l]) + _eps).sqrt().unsqueeze(0)
                if z_pre[l].dim() == 4:
                    mu[l] = mu[l].unsqueeze(2).unsqueeze(3)
                    sigma[l] = sigma[l].unsqueeze(2).unsqueeze(3)

            z[l] = self.batchnorm(self.W[l](h[l - 1]), l, 'clean')

            _beta = self.beta[l].unsqueeze(0)
            _gamma = self.gamma[l].unsqueeze(0)
            if z_tilde[l].dim() == 4:
                _beta = _beta.unsqueeze(2).unsqueeze(3)
                _gamma = _gamma.unsqueeze(2).unsqueeze(3)
            _beta = _beta.expand_as(z[l])
            _gamma = _gamma.expand_as(z[l])

            h[l] = self.act[l](_gamma * (z[l] + _beta))
            #print 'h', l, var_to_numpy(h[l]).max(), var_to_numpy(h[l]).min()
            assert not anynan(h[l].data)

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
            #print 'z_hat', l, var_to_numpy(z_hat[l]).max(), var_to_numpy(z_hat[l]).min()
            #print 'z_hat_bn', l, var_to_numpy(z_hat_bn[l]).max(), var_to_numpy(z_hat_bn[l]).min()
            assert not anynan(z_hat_bn[l].data)

        rec_loss = 0
        for l in range(0, self.L + 1):
            rec_loss += self.lambda_[l] * ((z[l] - z_hat_bn[l]) ** 2).mean()

        return y_tilde, y, rec_loss


lambdas = []
model = Ensemble()

with open('ladder.json') as config_file:
    config = json.load(config_file)

if args.lambdas is not None:
    lambdas = NP.loadtxt(args.lambdas)
    if lambdas.ndim == 1:
        lambdas = NP.expand_dims(lambdas, 0)
    lambdas = lambdas.tolist()
else:
    lambda_choices = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    for i in range(1 if args.ensemble is None else args.ensemble):
        if args.gamma:
            l = NP.concatenate([NP.zeros(10), [RNG.choice(lambda_choices)]])
        lambdas.append(RNG.choice(lambda_choices, len(config['layers']) + 2).tolist())

for i in range(1 if args.ensemble is None else args.ensemble):
    model.append(Ladder(config, lambdas[i]))

if args.cuda:
    model.cuda()

if (args.modelname is not None) and (args.lambdas is None):
    NP.savetxt('%s.cfg' % args.modelname, lambdas)
six.print_('Lambda: %s' % lambdas)
opt = OPT.Adam(model.parameters(), lr=1e-3)
best_acc = 0

def forward(data):
    global model
    y_tilde, y, rec_loss = model(data)
    return y_tilde.mean(0).squeeze(0), y.mean(0).squeeze(0), rec_loss.mean(0).squeeze(0)


def train_model():
    global best_acc
    for E in range(0, 500):
        model.train()
        for B in range(0, 600 if args.unlabeled else 30):
            (data_l, target_l), (data_u, target_u) = fetch()
            if args.unlabeled:
                data = T.cat([data_l, data_u])
                target = T.cat([target_l, target_u])
            else:
                data = data_l
                target = target_l
            data = (variable(data).float() / 255.).view(-1, 1, 28, 28)
            target = variable(target)
            opt.zero_grad()
            y_tilde, _, rec_loss = forward(data)

            labeled_mask = (target != -1).unsqueeze(1)
            y_tilde_labeled = y_tilde * labeled_mask.float().expand_as(y_tilde)
            target = target * labeled_mask.long()
            labeled_loss = F.nll_loss(y_tilde_labeled, target)

            loss = labeled_loss + rec_loss
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
            data = (variable(data, volatile=True).float() / 255.).view(-1, 1, 28, 28)
            target = variable(target, volatile=True)
            y_tilde, y, rec_loss = forward(data)
            assert not anynan(y_tilde.data) and not anynan(y.data) and not anynan(rec_loss.data)
            loss = F.nll_loss(y, target)
            valid_loss += loss.data[0] * data.size()[0]
            acc += (var_to_numpy(y).argmax(axis=1) == var_to_numpy(target)).sum()
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
