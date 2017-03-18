
from torch.optim import Optimizer

class RMSprop2(Optimizer):
    def __init__(
            self,
            params,
            aleph=0.95,
            beth=0.9,
            gimmel=1e-4,
            daleth=1e-4,
            weight_decay=0
            ):
        defaults = dict(
                aleph=aleph,
                beth=beth,
                gimmel=gimmel,
                daleth=daleth,
                weight_decay=0
                )
        super(RMSprop2, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['t'] = 0
                    state['n'] = grad.new().resize_as_(grad).zero_()
                    state['g'] = grad.new().resize_as_(grad).zero_()
                    state['delta'] = grad.new().resize_as_(grad).zero_()

                n = state['n']
                g = state['g']
                delta = state['delta']
                state['t'] += 1

                aleph = group['aleph']
                beth = group['beth']
                gimmel = group['gimmel']
                daleth = group['daleth']
                weight_decay = group['weight_decay']

                if weight_decay != 0:
                    grad = grad.add(weight_decay, p.data)

                n.mul_(aleph).addcmul_(1 - aleph, grad, grad)
                g.mul_(aleph).add_(1 - aleph, grad)
                grad_mod = grad / (n - g ** 2 + daleth).sqrt()
                delta.mul_(beth).add_(-gimmel, grad_mod)
                p.data.add_(delta)
        return loss
