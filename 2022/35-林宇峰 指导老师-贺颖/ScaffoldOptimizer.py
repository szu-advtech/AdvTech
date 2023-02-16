from torch.optim import Optimizer
import torch.optim
class _ScaffoldOptimizer(torch.optim.Adam):
    def __init__(self, params, lr, weight_decay):
        super().__init__(params,lr = lr)
    def step(self, server_controls, client_controls, closure = None):
        #if True:
        if server_controls is None or client_controls is None:
            super().step()
        else:
            for group in self.param_groups:
                for p, c, ci in zip(group['params'], server_controls, client_controls):
                    p.grad.data += c.data - ci.data
            super().step()

class ScaffoldOptimizer(Optimizer):
    def __init__(self, params, lr, weight_decay):
        default = dict(lr = lr, weight_decay = weight_decay)
        super().__init__(params, default)

    def step(self, server_controls, client_controls, closure = None):
        loss = None
        if closure is not None:
            loss = closure
        for group in self.param_groups:
            if True:
            #if server_controls is None or client_controls is None:
                for p in group['params']:
                    p.data = p.data - p.grad.data * group['lr']
            else:
                for p, c, ci in zip(group['params'], server_controls, client_controls):
                    p.grad.data += c.data - ci.data
                    p.data = p.data - p.grad.data*group['lr']




