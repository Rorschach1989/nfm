import torch
import torch.nn as nn


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


class TransNLL(nn.Module):
    """The negative log-likelihood of some transformation models,
    this module takes FULL-BATCH data, stochastic training is NOT supported"""

    def __init__(self, eps_conf: EpsDistribution, num_jumps):
        super(TransNLL, self).__init__()
        self.eps_conf = eps_conf
        # **Notes**: ideally, num_jumps shall not be known in advance, somewhat methodological issue
        # regarding failure of stochastic training of NPMLE
        self.log_jump_sizes = nn.Parameter(
            torch.log(1e-3 * torch.ones([num_jumps], dtype=torch.float)), requires_grad=True)

    def forward(self, m_z, y, delta):
        """Compute the negative log-likelihood given observed data

        Args:
            m_z(torch.Tensor): a tensor of shape [sample_size, 1]
            y(torch.Tensor): Y_i = T_i \wedge C_i, with shape [sample_size, 1]
            delta(torch.Tensor): \Delta_i = 1_{T_i \le C_i}, with shape [sample_size, 1]

        Returns:
            negative log-likelihood tensor of shape [1,]
        """
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

