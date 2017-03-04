
import torch as T
from torch.autograd import Variable
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--unlabeled",
                    help="use unlabeled data",
                    action="store_true")
parser.add_argument("--modelname",
                    help="model name",
                    type=str)
parser.add_argument("--loadmodel",
                    help="file to load model from for continuation",
                    type=str)
parser.add_argument("--lambdas",
                    help="file to load lambdas from",
                    type=str)
parser.add_argument("--cuda",
                    help="use cuda",
                    action="store_true")
parser.add_argument("--ensemble",
                    help="number of models",
                    type=int)
parser.add_argument("--gamma",
                    help="gamma model",
                    action="store_true")

args = parser.parse_args()

def variable(*args_, **kwargs):
    if args.cuda:
        return Variable(*args_, **kwargs).cuda()
    else:
        return Variable(*args_, **kwargs)

def var_to_numpy(v):
    if args.cuda:
        return v.cpu().data.numpy()
    else:
        return v.data.numpy()

def alloc_list(n):
    return [None] * n

def anynan(t):
    return ((t != t).sum() > 0) or ((t.abs() > 1e+7).sum() > 0)

def noise(size, scale=0.3, center=0):
    return variable(T.randn(*size)) * scale + center

def noised(x, scale=0.3, center=0):
    return x + variable(T.randn(*x.size())) * scale + center

def batchnorm_mean_var(x):
    ndim = x.dim()
    batch_size = x.size()[0]

    if ndim == 4:
        # Torch does not have mean/sum/variance over multiple axes...
        mean = x.mean(0).mean(2).mean(3)
        diff = x - mean.expand_as(x)
        var = (diff ** 2).mean(0).mean(2).mean(3)
    else:
        mean = x.mean(0)
        var = x.var(0) * (batch_size - 1) / batch_size

    return mean, var
