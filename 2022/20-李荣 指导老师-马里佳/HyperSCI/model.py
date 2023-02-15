import torch
from torch import Tensor
from torch_scatter import scatter_add
from torch_geometric.nn.dense.linear import Linear
from torch.nn import LeakyReLU, MSELoss
from torch_geometric.nn import HypergraphConv, MLP
from torch.distributions import bernoulli, normal
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def init_weights(m):
    if type(m) == Linear:
        torch.nn.init.normal_(m.weight, mean=0, std=0.01)
        m.bias.data.zero_()

class SinkhornDistance(torch.nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, p=0.5, max_iter=10, lam=10):
        super(SinkhornDistance, self).__init__()
        self.p = p
        self.max_iter = max_iter
        self.lam = lam

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        nx = x.shape[0]
        ny = y.shape[0]

        x = x.squeeze()
        y = y.squeeze()

        #    pdist = torch.nn.PairwiseDistance(p=2)

        M = self._cost_matrix(x, y)  # distance_matrix(x,y,p=2)

        '''estimate lambda and delta'''
        M_mean = torch.mean(M)
        M_drop = F.dropout(M, 10.0 / (nx * ny))
        delta = torch.max(M_drop).cpu().detach()
        eff_lam = (self.lam / M_mean).cpu().detach()

        '''compute new distance matrix'''
        Mt = M
        row = delta * torch.ones(M[0:1, :].shape)
        col = torch.cat([delta * torch.ones(M[:, 0:1].shape), torch.zeros((1, 1))], 0)
        row = row.to(device)
        col = col.to(device)
        Mt = torch.cat([M, row], 0)
        Mt = torch.cat([Mt, col], 1)

        '''compute marginal'''
        a = torch.cat([self.p * torch.ones((nx, 1)) / nx, (1 - self.p) * torch.ones((1, 1))], 0)
        b = torch.cat([(1 - self.p) * torch.ones((ny, 1)) / ny, self.p * torch.ones((1, 1))], 0)

        '''compute kernel'''
        Mlam = eff_lam * Mt
        temp_term = torch.ones(1) * 1e-6
        temp_term = temp_term.to(device)
        a = a.to(device)
        b = b.to(device)
        K = torch.exp(-Mlam) + temp_term
        U = K * Mt
        ainvK = K / a

        u = a

        for i in range(self.max_iter):
            u = 1.0 / (ainvK.matmul(b / torch.t(torch.t(u).matmul(K))))
            u = u.to(device)
        v = b / (torch.t(torch.t(u).matmul(K)))
        v = v.to(device)

        upper_t = u * (torch.t(v) * K).detach()

        E = upper_t * Mt
        D = 2 * torch.sum(E)

        D = D.to(device)

        return D, Mlam

    @staticmethod
    def _cost_matrix(x, y, p=2, eps=1e-5):
        "Returns the matrix of $|x_i-y_j|^p$."
        n_1, n_2, dim = x.size(0), y.size(0), x.size(-1)
        expanded_1 = x.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = y.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** p
        C = torch.sum(differences, dim=2, keepdim=False)
        return (eps + C) ** (1. / p)

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

class LossFunc(torch.nn.Module):
    def __init__(self, wass):
        super(LossFunc, self).__init__()
        self.alpha = wass
        self.loss_mse = MSELoss(reduction='sum')
        self.num_balance_max = 2000

    def forward(self, Y_true, treatments, results, idx_trn, idx_select):
        '''standard mean squared error'''
        # binary
        y1_true = Y_true[1]
        y0_true = Y_true[0]
        rep = results['p']
        y1_pred = results['y1_pred'].squeeze()
        y0_pred = results['y0_pred'].squeeze()
        yf_pred = torch.where(treatments > 0, y1_pred, y0_pred)

        # balancing
        idx_balance = idx_select if len(idx_select) < self.num_balance_max else idx_select[: self.num_balance_max]

        rep_t1, rep_t0 = rep[idx_balance][(treatments[idx_balance] > 0).nonzero()], rep[idx_balance][
            (treatments[idx_balance] < 1).nonzero()]

        # wass1 distance
        wasserstein = SinkhornDistance()
        dist, _ = wasserstein(rep_t0, rep_t1)

        # potential outcome prediction
        YF = torch.where(treatments > 0, y1_true, y0_true)

        # norm y
        ym, ys = torch.mean(YF[idx_trn]), torch.std(YF[idx_trn])
        YF_select = (YF[idx_select] - ym) / ys

        # loss: (Y-Y_hat)^2 + alpha * w-dist
        loss_y = self.loss_mse(yf_pred[idx_select], YF_select)

        loss = loss_y + self.alpha * dist

        loss_result = {
            'loss': loss, 'loss_y': loss_y, 'loss_b': dist
        }
        return loss_result

class HyperSCI(torch.nn.Module):
    def __init__(self, n_in, args):
        super().__init__()
        self.confusion_model = MLP([n_in, args.h_dim, args.h_dim], dropout=args.dropout)
        self.interference_model = HypergraphConv(in_channels=args.h_dim,
                                                 out_channels=args.g_dim,
                                                 use_attention=True,
                                                 heads = 2, dropout=args.dropout,
                                                 concat=False)
        y_pred_dim = args.h_dim + args.g_dim
        self.f0_model = MLP([y_pred_dim, y_pred_dim, 1], dropout=args.dropout)
        self.f1_model = MLP([y_pred_dim, y_pred_dim, 1], dropout=args.dropout)
        self.leaklyrelu = LeakyReLU()

        '''
        self.confusion_model.apply(lambda m:init_weights(m))
        self.interference_model.apply(lambda m:init_weights(m))
        self.f0_model.apply(lambda m:init_weights(m))
        self.f1_model.apply(lambda m:init_weights(m))
        '''

    def forward(self, X: Tensor, T: Tensor, hyperedge_index: Tensor) -> dict:
        results = {}

        Z = self.confusion_model(X).to(device)

        '''confusion balance loss'''

        T = T.reshape((-1,1))
        P = Z * T
        P.to(device)

        hyperedge_attr = scatter_add(P[hyperedge_index[0]],
                                     hyperedge_index[1], dim=0).to(device)
        B = scatter_add(P.new_ones(hyperedge_index.size(1)),
                        hyperedge_index[1], dim=0).reshape(-1,1).to(device)
        hyperedge_attr = hyperedge_attr / B

        P = self.interference_model(x=P, hyperedge_index=hyperedge_index, hyperedge_attr=hyperedge_attr)


        P = torch.cat([Z, self.leaklyrelu(P)], dim=1)

        y0 = self.f0_model(P)
        y1 = self.f1_model(P)

        results['y0_pred'] = y0
        results['y1_pred'] = y1
        results['p'] = P

        return results


class cfrnet(torch.nn.Module):
    def __init__(self, n_in, args):
        super().__init__()
        self.confusion_model = MLP([n_in, args.h_dim, args.h_dim], dropout=args.dropout)
        self.f0_model = MLP([args.h_dim, args.h_dim, 1], dropout=args.dropout)
        self.f1_model = MLP([args.h_dim, args.h_dim, 1], dropout=args.dropout)
        self.leaklyrelu = LeakyReLU()

    def forward(self, X: Tensor, T: Tensor, hyperedge_index: Tensor) -> dict:
        results = {}

        Z = self.confusion_model(X).to(device)

        '''confusion balance loss'''

        T = T.reshape((-1,1))

        y0 = self.f0_model(Z)
        y1 = self.f1_model(Z)

        results['y0_pred'] = y0
        results['y1_pred'] = y1
        results['p'] = Z

        return results





