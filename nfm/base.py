import torch
import torch.nn as nn
from .monotone import MonotoneMLP
from .umnn import UMNN
from .umnn_v2 import ParallelNeuralIntegral, _flatten
from .utils import default_device


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

    def g(self, x):
        pass

    def log_g_derivative(self, x):
        pass


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
        self.to(default_device)

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
        """Survival prediction taking the shape N * T, with N being sample size
        semantics shall be: m[i, j] denotes the survival prediction of individual j at ordered time t_i
        """
        lambda_arg = torch.log(self.get_transform_prediction(y_test)) + m_z.view(1, -1)
        shape = lambda_arg.shape
        return self.eps_conf.survival(lambda_arg.view(-1, 1)).view(shape)

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
        total_jumps = torch.zeros([sample_size], dtype=torch.float, device=default_device)
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
    """

    def __init__(self, eps_conf: EpsDistribution, num_hidden_units, **kwargs):
        super(MonotoneNLL, self).__init__()
        self.eps_conf = eps_conf
        self.h = UMNN(num_hidden_units=num_hidden_units, **kwargs)
        self.to(default_device)

    def get_survival_prediction(self, m_z, y_test):
        lambda_arg = torch.log(self.h(y_test)) + m_z.view(1, -1)
        return self.eps_conf.survival(lambda_arg)

    def forward(self, m_z, y, delta):
        """c.f. NPMLENLL, the calculation is way more direct"""
        uncensored = torch.where(delta)[0]
        batch_size = y.shape[0]
        # h_y = self.h(y)
        # h_derive_y = self.h.get_derivative(y)
        h_y_ = self.h(y)
        h_y = torch.log(h_y_ + 1e-15)
        h_derive_y = self.h.get_derivative(y) / (h_y_ + 1e-15)
        lambda_arg = m_z + h_y
        surv_part = self.eps_conf.cumulative_hazard(lambda_arg).sum()
        intensity_part = - torch.log(h_derive_y[uncensored] + 1e-15).sum() \
                         - self.eps_conf.log_hazard(lambda_arg[uncensored]).sum()
        return (surv_part + intensity_part) / batch_size


class CSNLL(MonotoneNLL):
    """Use current status paradigm"""

    def __init__(self, eps_conf: EpsDistribution, num_hidden_units, **kwargs):
        super(CSNLL, self).__init__(eps_conf, num_hidden_units, **kwargs)
        self.nll = nn.NLLLoss()

    def forward(self, m_z, y, delta):
        delta = delta.view(-1)
        h_y_ = self.h(y)
        h_y = torch.log(h_y_ + 1e-15)
        lambda_arg = m_z + h_y
        cum_h = self.eps_conf.cumulative_hazard(lambda_arg)
        logits = torch.cat([-cum_h, torch.log(1 - torch.exp(-cum_h))], dim=1)
        return self.nll(logits, delta.type(torch.long))


class FullyNeuralNLL(nn.Module):
    """A more general model with
    \lambda(t | Z) = e^{\mu(t, Z)}
    with \mu provided outside
    """

    def __init__(self, eps_conf: EpsDistribution, encoder: nn.Module, nb_steps=20):
        super(FullyNeuralNLL, self).__init__()
        self.eps_conf = eps_conf
        self.encoder = encoder  # encoder shall be a positive map
        self.nb_steps = nb_steps

    def get_survival_prediction(self, z_test, y_test):
        batch_size = z_test.shape[0]
        n_times = y_test.shape[0]
        z_test_ = torch.tile(z_test, [n_times]).view(batch_size * n_times, -1)
        y_test_ = torch.tile(y_test, [batch_size, 1]).view(batch_size * n_times, 1)
        _, cum_hazard = self.get_cumulative_hazard(z_test_, y_test_)
        return torch.exp(- cum_hazard).view(batch_size, n_times).T

    def get_cumulative_hazard(self, z, y):
        shape = y.shape
        y = y.view(-1, 1)
        z = z.reshape(y.shape[0], -1)
        y0 = torch.zeros(y.shape).to(default_device)
        int_encoder = ParallelNeuralIntegral.apply(
            y0, y, self.encoder, _flatten(self.encoder.parameters()), z, self.nb_steps
        ).view(shape)
        return int_encoder, self.eps_conf.g(int_encoder)

    def forward(self, z, y, delta):
        uncensored = torch.where(delta)[0]
        batch_size = y.shape[0]
        int_encoder, cum_hazard = self.get_cumulative_hazard(z, y)
        log_hazard = torch.log(self.encoder(y, z)) + self.eps_conf.log_g_derivative(int_encoder)
        return (- log_hazard[uncensored].sum() + cum_hazard.sum()) / batch_size

