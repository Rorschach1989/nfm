import torch
import torch.nn as nn
from .monotone import MonotoneMLP


class EpsDistribution(object):
    """Abstraction of (univariate) distribution of epsilon, shall be differentiable and vectorized
    Currently we don't allow the configuration to contain learnable parameters, per original definition
    of transformation models
    """

    def hazard(self, x):
        """Hazard function"""
        raise NotImplementedError

    def cumulative_hazard(self, x):
        """Cumulative hazard function"""
        raise NotImplementedError

    def log_hazard(self, x):
        """Log hazard function, provide explicitly if possible (avoiding certain numerical issues)"""
        return torch.log(self.hazard(x))

    def survival(self, x):
        return torch.exp(-self.cumulative_hazard(x))


class NPMLENLL(nn.Module):
    """The negative log-likelihood of some transformation models,
    this module takes FULL-BATCH data, stochastic training is NOT supported"""

    def __init__(self, eps_conf: EpsDistribution, num_jumps):
        super(NPMLENLL, self).__init__()
        self.eps_conf = eps_conf
        # **Notes**: ideally, num_jumps shall not be known in advance, somewhat methodological issue
        # regarding failure of stochastic training of NPMLE
        self.log_jump_sizes = nn.Parameter(
            torch.log(1e-3 * torch.ones([num_jumps], dtype=torch.float)), requires_grad=True)
        self._anchor: torch.Tensor = None

    def _record_anchor(self, y, delta):
        if self._anchor is None:  # No checks for anchor incoherence
            self._anchor = y.masked_select(delta.type(torch.bool)).view(-1)

    def get_transform_prediction(self, y_test):  # Presumably run in no_grad mode
        assert self._anchor is not None
        step_fn = torch.cumsum(torch.exp(self.log_jump_sizes), dim=0)
        idxes = torch.minimum(torch.searchsorted(self._anchor, y_test.view(-1), right=True),
                              torch.as_tensor(self.log_jump_sizes.shape[0] - 1))
        return step_fn[idxes].view(-1, 1)

    def get_survival_prediction(self, m_z, y_test):  # Presumably run in no_grad mode
        lambda_arg = torch.log(self.get_transform_prediction(y_test)) + m_z.view(1, -1)
        return self.eps_conf.survival(lambda_arg)

    def forward(self, m_z, y, delta):
        """Compute the negative log-likelihood given observed data, under log transform
        \log H(T) = - \beta^T Z + \epsilon

        Args:
            m_z(torch.Tensor): a tensor of shape [sample_size, 1]
            y(torch.Tensor): Y_i = T_i \wedge C_i, with shape [sample_size, 1]
            delta(torch.Tensor): \Delta_i = 1_{T_i \le C_i}, with shape [sample_size, 1]

        Returns:
            negative log-likelihood tensor of shape [1,]
        """
        self._record_anchor(y, delta)
        sample_size = y.shape[0]
        total_jumps = torch.zeros([sample_size], dtype=torch.float)
        uncensored = torch.where(delta)[0]
        total_jumps[uncensored] = torch.exp(self.log_jump_sizes)
        log_h = torch.log(torch.cumsum(total_jumps, dim=0) + 1e-15).reshape([-1, 1])
        lambda_arg = log_h + m_z
        surv_part = self.eps_conf.cumulative_hazard(lambda_arg).sum()
        intensity_part = - self.eps_conf.log_hazard(lambda_arg[uncensored]).sum()\
                         - self.log_jump_sizes.sum() + log_h[uncensored].sum()
        return (surv_part + intensity_part) / sample_size


TransNLL = NPMLENLL  # Default transformation model NLL


class MonotoneNLL(nn.Module):
    """Transformation model NLL using monotone approximation of transformation function
    To save from unnecessary complexity, here we do NOT use log-transformed h, i.e,
    H(T) = -\beta^T Z + \epsilon
    """

    def __init__(self, eps_conf: EpsDistribution, num_hidden_units):
        super(MonotoneNLL, self).__init__()
        self.eps_conf = eps_conf
        self.h = MonotoneMLP(num_hidden_units=num_hidden_units)

    def forward(self, m_z, y, delta):
        """c.f. NPMLENLL, the calculation is way more direct"""
        uncensored = torch.where(delta)[0]
        batch_size = y.shape[0]
        # h_y = self.h(y)
        # h_derive_y = self.h.get_derivative(y)
        h_y_ = self.h(y)
        h_y = torch.log(h_y_)
        h_derive_y = self.h.get_derivative(y) / (h_y_ + 1e-15)
        lambda_arg = m_z + h_y
        surv_part = self.eps_conf.cumulative_hazard(lambda_arg).sum()
        intensity_part = - torch.log(h_derive_y[uncensored]).sum() \
                         - self.eps_conf.log_hazard(lambda_arg[uncensored]).sum()
        return (surv_part + intensity_part) / batch_size
