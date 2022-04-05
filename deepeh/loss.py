import math
import torch
import torch.nn as nn
from deeptrm.umnn import integrate


class DehLoss(nn.Module):
    """Loss function of the deep extended hazard model, adapted from original author's impl"""

    def __init__(self, kernel='gaussian', nb_steps=20):
        super(DehLoss, self).__init__()
        self._g1: torch.Tensor = None
        self._g2: torch.Tensor = None
        self._r: torch.Tensor = None
        self._h: torch.Tensor = None
        self._delta: torch.Tensor = None
        if kernel == 'gaussian':
            self.kernel = torch.distributions.Normal(0., 1.)
        else:
            raise NotImplementedError
        self.nb_steps = nb_steps

    def _set_anchor(self, g1, g2, r, delta):
        self._g1, self._g2, self._r, self._delta = g1, g2, r, delta

    def _set_bandwidth(self, h):
        if self._h is None:
            self._h = h

    @property
    def bandwidth(self):
        return self._h

    def get_baseline_hazard(self, y_test):
        kernel_arg = (self._r.view(-1, 1) - torch.log(y_test.view(1, -1))) / self.bandwidth
        coef = torch.exp(self._g2 - self._g1)
        nom = torch.mean(torch.exp(self.kernel.log_prob(kernel_arg)) * self._delta / self.bandwidth, dim=0)
        denom = torch.mean(self.kernel.cdf(kernel_arg) * coef, dim=0)
        # [t, 1] output
        return (nom / denom).view(-1, 1) / y_test.view(-1, 1)

    def _get_survival_prediction(self, m_z, y_test):
        g1 = m_z[:, 0]
        g2 = m_z[:, 1]
        n_times = y_test.shape[0]
        lambda_arg = y_test.view(-1, 1) * torch.exp(g1).view(1, -1)
        # Use Clenshaw-Curtis quadrature to compute integral of the messy baseline hazard estimate
        lambda_arg = lambda_arg.view(-1, 1)  # integrate method accepts no multi-dimension ones
        x0 = torch.zeros_like(lambda_arg) + 1e-15
        baseline_surv = integrate(x0, self.nb_steps, (lambda_arg - x0)/self.nb_steps, self.get_baseline_hazard, False)
        baseline_surv = baseline_surv.view(n_times, -1)
        return torch.exp(- baseline_surv * torch.exp(g2 - g1).view(-1))

    def get_survival_prediction(self, m_z, y_test, use_loop=True):
        if not use_loop:
            return self._get_survival_prediction(m_z, y_test)
        else:  # intermediate memory footprint is unacceptable sometimes
            preds = []
            n_times = y_test.shape[0]
            for i in range(n_times):
                preds.append(self._get_survival_prediction(m_z, y_test[i]))
            return torch.cat(preds, dim=0)

    def forward(self, m_z, y, delta):
        n = m_z.shape[0]
        g1 = m_z[:, 0].view(-1, 1)
        g2 = m_z[:, 1].view(-1, 1)
        h = 1.30 * math.pow(n, -0.2)
        delta = delta.view(n, 1)

        # R = g(Xi) + log(Oi)
        R = torch.add(g1, torch.log(y))
        self._set_anchor(g1, g2, R, delta)
        self._set_bandwidth(h)
        S1 = (delta * g2).sum() / n
        S2 = -(delta * R).sum() / n

        # Rj - Ri
        rawones = torch.ones(1, n)
        R1 = torch.mm(R, rawones)
        R2 = torch.mm(torch.t(rawones), torch.t(R))
        DR = R1 - R2

        # K[(Rj-Ri)/h]
        K = torch.exp(self.kernel.log_prob(DR / h))
        Del = torch.mm(delta, rawones)
        DelK = Del * K

        # (1/nh) *sum_j Deltaj * K[(Rj-Ri)/h]
        Dk = torch.sum(DelK, dim=0) / (n * h)  ## Dk would be zero as learning rate too large!

        # log {(1/nh) * Deltaj * K[(Rj-Ri)/h]}
        log_Dk = torch.log(Dk + 1e-15)

        S3 = (torch.t(delta) * log_Dk).sum() / n

        # Phi((Rj-Ri)/h)
        P = self.kernel.cdf(DR / h)
        L = torch.exp(g2 - g1)
        LL = torch.mm(L, rawones)
        LP_sum = torch.sum(LL * P, dim=0) / n
        Q = torch.log(LP_sum + 1e-15)

        S4 = -(delta * Q.view(n, 1)).sum() / n

        S = S1 + S2 + S3 + S4
        return -S

