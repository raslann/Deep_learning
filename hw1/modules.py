import torch as T
import torch.nn as NN


class BufferList(NN.Module):
    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self += buffers

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        return self._buffers[str(idx)]

    def __setitem__(self, idx, buf):
        return self.register_buffer(str(idx), buf)

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())

    def __iadd__(self, buffers):
        return self.extend(buffers)

    def append(self, buf):
        self.register_buffer(str(len(self)), buf)
        return self

    def extend(self, buffers):
        if not isinstance(buffers, list):
            raise TypeError("ParameterList.extend should be called with a "
                            "list, but got " + type(buffers).__name__)
        offset = len(self)
        for i, buf in enumerate(buffers):
            self.register_buffer(str(offset + i), buf)
        return self


class GlobalAvgPool2d(NN.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return x.view(x.size()[0], x.size()[1], -1).mean(2).squeeze(2)


class GlobalUpsample2d(NN.Module):
    def __init__(self, size):
        super(GlobalUpsample2d, self).__init__()
        self.size = size

    def forward(self, x):
        return (x
                .view(x.size()[0], x.size()[1], 1, 1)
                .expand(x.size()[0], x.size()[1], self.size, self.size))


class Ensemble(NN.ModuleList):
    def __init__(self, modules=None):
        super(Ensemble, self).__init__()

    def forward(self, *args):
        result = []
        for i in range(len(self)):
            r = self[i].forward(*args)
            if len(result) == 0:
                for i in range(len(r)):
                    result.append([])
            for i in range(len(r)):
                result[i].append(r[i])

        for i in range(len(result)):
            result[i] = T.stack(result[i])
        return result
