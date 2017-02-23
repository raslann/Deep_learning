
import torch as T
from torch.autograd import Variable

def alloc_list(n):
    return [None] * n

def anynan(t):
    return (t != t).sum() > 0

def noise(size, scale=0.3, center=0):
    return Variable(T.randn(*size)) * scale + center

def noised(x, scale=0.3, center=0):
    return x + Variable(T.randn(*x.size())) * scale + center
