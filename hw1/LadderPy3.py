
import torch as T
from torch.autograd import Variable
import torch.nn as NN
import torch.nn.functional as F
from torch.nn import Parameter
import torch.optim as OPT
import numpy as NP

_eps = 1e-8

def alloc_list(n):
    return [None] * n

def anynan(t):
    return (t != t).sum() > 0

class Denoiser(NN.Module):
    def __init__(self, *size):
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
        self.a10 = Parameter(T.ones(size))

    def forward(self, z, u):
        assert z.size() == u.size()

        a = [None] * 11
        for i in range(1, 11):
            a[i] = getattr(self, "a%d" % i).unsqueeze(0).expand_as(z)

        mu = a[1] * F.sigmoid(a[2] * u + a[3]) + a[4] * u + a[5]
        nu = a[6] * F.sigmoid(a[7] * u + a[8]) + a[9] * u + a[10]

        return (z - mu) * nu + mu


def noise(size, scale=0.01, center=0):
    return Variable(T.randn(*size)) * scale + center


def noised(x, scale=0.01, center=0):
    return x + Variable(T.randn(*x.size())) * scale + center


class Ladder(NN.Module):
    def _add_W(self, input_config, output_config):
        l = NN.Linear(input_config, output_config)
        self.add_module('W%d' % len(self.W), l)
        self.W.append(l)

    def _add_act(self, act_module):
        self.add_module('A%d' % len(self.act), act_module)
        self.act.append(act_module)

    def _add_V(self, input_config, output_config):
        l = NN.Linear(input_config, output_config)
        self.add_module('V%d' % (self.L - len(self.V) + 1), l)
        self.V.append(l)

    def _add_g(self, state_config):
        d = Denoiser(state_config)
        self.add_module('g%d' % len(self.g), d)
        self.g.append(d)

    def _add_stats(self, state_config, init_scale=1):
        gamma = Parameter(T.randn(state_config) * init_scale, requires_grad=True)
        beta = Parameter(T.zeros(state_config), requires_grad=True)
        mean = Variable(T.zeros(state_config))
        var = Variable(T.zeros(state_config))
        mean_noisy = Variable(T.zeros(state_config))
        var_noisy = Variable(T.zeros(state_config))
        mean_dec = Variable(T.zeros(state_config))
        var_dec = Variable(T.zeros(state_config))

        self.register_parameter('gamma%d' % len(self.gamma), gamma)
        self.register_parameter('beta%d' % len(self.beta), beta)

        self.gamma.append(gamma)
        self.beta.append(beta)
        self.mean.append(mean)
        self.var.append(var)
        self.mean_noisy.append(mean_noisy)
        self.var_noisy.append(var_noisy)
        self.mean_dec.append(mean_dec)
        self.var_dec.append(var_dec)

    def _add_input_stats(self, input_config):
        self.mean_noisy.insert(0, Variable(T.zeros(input_config)))
        self.var_noisy.insert(0, Variable(T.zeros(input_config)))
        self.mean.insert(0, Variable(T.zeros(input_config)))
        self.var.insert(0, Variable(T.zeros(input_config)))
        self.mean_dec.insert(0, Variable(T.zeros(input_config)))
        self.var_dec.insert(0, Variable(T.zeros(input_config)))

    def __init__(self, layers, lambda_):
        '''
        Parameters
        ----------
        layers : list: [input_size, hidden_size1, ..., output_size]
        '''
        super(Ladder, self).__init__()

        self.L = len(layers) - 1
        self.lambda_ = lambda_

        self.W = [None]
        self.act = [None]
        self.V = [None]
        self.g = []
        self.gamma = [None]
        self.beta = [None]
        self.mean_noisy = []
        self.var_noisy = []
        self.mean = []
        self.var = []
        self.mean_dec = []
        self.var_dec = []
        self.count = 0

        self._add_g(layers[0])    # g_0
        for l in range(1, self.L + 1):
            self._add_W(layers[l - 1], layers[l])
            self._add_act(NN.ReLU() if l != self.L else NN.LogSoftmax())
            self._add_V(layers[l], layers[l - 1])
            self._add_g(layers[l])
            self._add_stats(layers[l])
        self._add_input_stats(layers[0])

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

        h_tilde[0] = z_tilde[0] = noised(self.batchnorm(x, 0, 'noisy'))

        for l in range(1, self.L + 1):
            z_tilde[l] = noised(self.batchnorm(self.W[l](h_tilde[l - 1]), l, 'noisy'))
            _beta = self.beta[l].unsqueeze(0).expand_as(z_tilde[l])
            _gamma = self.gamma[l].unsqueeze(0).expand_as(z_tilde[l])
            h_tilde[l] = self.act[l](_gamma * (z_tilde[l] + _beta))

        y_tilde = h_tilde[self.L]

        h[0] = z[0] = self.batchnorm(x, 0, 'clean')
        mu[0] = z[0].mean(0)
        sigma[0] = (z[0].var(0) * (batch_size - 1) / batch_size + _eps).sqrt()

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
            _mu = mu[l].expand_as(z_hat[l])
            _sigma = sigma[l].expand_as(z_hat[l])
            assert (_sigma.data == 0).sum() == 0
            z_hat_bn[l] = (z_hat[l] - _mu) / _sigma

        rec_loss = 0
        for l in range(0, self.L + 1):
            rec_loss += self.lambda_[l] * ((z[l] - z_hat_bn[l]) ** 2).mean()

        return y_tilde, rec_loss


model = Ladder([784, 500, 300, 100, 10], [0.001, 0.001, 0.001, 0.001, 0.001])
opt = OPT.Adam(model.parameters(), lr=1e-3)
import pickle 

train_labeled = pickle.load(open("train_labeled.p", "rb")) 
train_unlabeled = pickle.load(open("train_unlabeled.p", "rb")) 
valid = pickle.load(open("validation.p", "rb")) 
'''
import cPickle
with open('train_labeled.p', 'rb') as f:
    train_labeled = cPickle.load(f)
with open('train_unlabeled.p', 'rb') as f:
    train_unlabeled = cPickle.load(f)
with open('validation.p', 'rb') as f:
    valid = cPickle.load(f)
'''

train_labeled_data = train_labeled.train_data
train_unlabeled_data = train_unlabeled.train_data
train_data = T.cat((train_labeled_data, train_unlabeled_data))
train_labeled_labels = train_labeled.train_labels
train_unlabeled_labels = T.zeros(train_unlabeled_data.size()[0]).long() - 1
train_labels = T.cat((train_labeled_labels, train_unlabeled_labels))

labeled_dataset = T.utils.data.TensorDataset(train_labeled_data, train_labeled_labels)
train_dataset = T.utils.data.TensorDataset(train_data, train_labels)
valid_dataset = T.utils.data.TensorDataset(valid.test_data, valid.test_labels)
labeled_loader = T.utils.data.DataLoader(labeled_dataset, 64, True)
train_loader = T.utils.data.DataLoader(train_dataset, 64, True)
valid_loader = T.utils.data.DataLoader(valid_dataset, 64, True)


def train_model():
    iteration = 0
    for E in range(0, 100):
        model.train()
        for B, (data, target) in enumerate(train_loader):
            iteration += 1
            data = (Variable(data).float() / 255.).view(-1, 28 * 28)
            target = Variable(target)
            opt.zero_grad()
            y_tilde, rec_loss = model(data)
            labeled_mask = (target != -1).unsqueeze(1)
            unlabeled_mask = (target > -1).unsqueeze(1)
            y_tilde = y_tilde * labeled_mask.float().expand_as(y_tilde)
            #y_tilde_unlabeled = y_tilde * unlabeled_mask.float().expand_as(y_tilde)
            target = target * labeled_mask.long()
            labeled_loss = F.nll_loss(y_tilde, target)
            psuedo_labeled_loss = y_tilde.max(1)[0]
            psuedo_labeled_loss = ((psuedo_labeled_loss * unlabeled_mask.float()) * \
                                   NP.min((.001, 1e-6*iteration))).mean()
            loss = labeled_loss + rec_loss*10 - psuedo_labeled_loss
            assert not anynan(loss.data)

            loss.backward()
            for p in model.parameters():
                assert not anynan(p.grad.data)
            opt.step()
            #print('#%05d      %.5f' % (B, loss.data[0]))

        model.eval()
        valid_loss = 0
        acc = 0
        for B, (data, target) in enumerate(valid_loader):
            data = (Variable(data, volatile=True).float() / 255.).view(-1, 28 * 28)
            target = Variable(target, volatile=True)
            y_tilde, rec_loss = model(data)
            loss = F.nll_loss(y_tilde, target)
            valid_loss += loss.data[0] * data.size()[0]
            acc += (y_tilde.data.numpy().argmax(axis=1) == target.data.numpy()).sum()
            del data, target, y_tilde
        valid_loss /= len(valid_loader.dataset)
        #acc /= 
        print('@%05d      %.5f (%d)' % (E, valid_loss, acc))
        print(len(valid_loader.dataset))

train_model()
