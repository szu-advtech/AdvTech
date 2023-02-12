import jittor.nn as nn
import jittor as jt
import jittor.init
class spectral_norm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1, eps=1e-12):
        super(spectral_norm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        self.eps = eps
        if not self._made_params():
            self._make_params()

    def l2normalize(self, v):
        return v / (v.norm() + self.eps)

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name)

        height = w.shape[0]
        for _ in range(self.power_iterations):
            v.assign(self.l2normalize((w.view(height,-1).t() * u.unsqueeze(0)).sum(-1)))
            u.assign(self.l2normalize((w.view(height,-1) * v.unsqueeze(0)).sum(-1)))
        sigma = (u * (w.view(height,-1) * v.unsqueeze(0)).sum(-1)).sum()
        getattr(self.module, self.name).assign(w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name)
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.shape[0]
        width = w.view(height, -1).shape[1]

        u = jt.empty([height], dtype=w.dtype).gauss_(0, 1)
        v = jt.empty([width], dtype=w.dtype).gauss_(0, 1)
        u = self.l2normalize(u)
        v = self.l2normalize(v)

        setattr(self.module, self.name + "_u", u.stop_grad())
        setattr(self.module, self.name + "_v", v.stop_grad())

    def execute(self, *args):
        self._update_u_v()
        return self.module.execute(*args)


