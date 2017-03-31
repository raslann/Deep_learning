
import torch as T
from torch.autograd import Variable
import argparse
import numpy as NP
import numpy.random as RNG

parser = argparse.ArgumentParser()
parser.add_argument("--modelname",
                    help="model name",
                    type=str)
parser.add_argument("--dropout",
                    help="dropout applied to layers",
                    type=float,
                    default=0.2)
parser.add_argument("--nlayers",
                    help="number of layers in the LSTM",
                    type=float,
                    default=2)
parser.add_argument("--tie_weight",
                    help="tie input and output weights",
                    action="store_true",
                    default=False)
parser.add_argument("--multi_LSTM",
                    help="Use multi layer LSTM",
                    action="store_true",
                    default=False)
parser.add_argument("--GRUCell",
                    help="Use GRU Cell",
                    action="store_true",
                    default=False)
parser.add_argument("--loadmodel",
                    help="file to load model from for continuation",
                    type=str)
parser.add_argument("--cuda",
                    help="use cuda",
                    action="store_true",
                    default=False)
parser.add_argument("--trainname",
                    help="training corpus name",
                    type=str)
parser.add_argument("--validname",
                    help="validation corpus name",
                    type=str)
parser.add_argument("--testname",
                    help="testing corpus name",
                    type=str)
parser.add_argument("--epochs",
                    help="number of epochs (default 200)",
                    type=int,
                    default=200)
parser.add_argument("--batchsize",
                    help="batch size (default 128)",
                    type=int,
                    default=32)
parser.add_argument("--embedsize",
                    help="embedding vector size (default 256)",
                    type=int,
                    default=650)
parser.add_argument("--statesize",
                    help="hidden state vector size (default 256)",
                    type=int,
                    default=650)
parser.add_argument("--minlength",
                    help="minimum sentence length (default 1)",
                    type=int,
                    default=1)
parser.add_argument("--gradnorm",
                    help="gradient clipping norm",
                    type=float,
                    default=0.02)

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

def anynan(t):
    return ((t != t).sum() > 0) or ((t.abs() > 1e+7).sum() > 0)


# Initializations from keras

def get_fans(shape):
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif (len(shape) == 4) or (len(shape) == 5):
        fan_in = NP.prod(shape[1:])
        fan_out = shape[0]
    else:
        fan_in = NP.sqrt(NP.prod(shape))
        fan_out = NP.sqrt(NP.prod(shape))

    return fan_in, fan_out

def glorot_uniform(shape):
    fan_in, fan_out = get_fans(shape)
    s = NP.sqrt(6. / (fan_in + fan_out))
    return RNG.uniform(low=-s, high=s, size=shape)

def orthogonal(shape, scale=1.1):
    flat_shape = (shape[0], NP.prod(shape[1]))
    a = RNG.normal(0, 1, flat_shape)
    u, _, v = NP.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return scale * q[:shape[0], :shape[1]]
