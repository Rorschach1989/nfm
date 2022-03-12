import torch
import torch.nn as nn
from .monotone import MonotoneMLP
from .base import EpsDistribution


class GaussianEps(EpsDistribution):
    """As-is"""

    def __init__(self):
        self._gaussian_rv = torch.distributions.Normal(0., 1.)

    def hazard(self, x):
        return torch.exp(self._gaussian_rv.log_prob(x)) / (1 - self._gaussian_rv.cdf(x))

    def log_hazard(self, x):
        return - torch.log((1 - self._gaussian_rv.cdf(x))) + self._gaussian_rv.log_prob(x)

    def cumulative_hazard(self, x):
        return - torch.log(1 - self._gaussian_rv.cdf(x))


class CoxEps(EpsDistribution):
    """As-is"""

    def hazard(self, x):
        return torch.exp(x)

    def log_hazard(self, x):
        return x

    def cumulative_hazard(self, x):
        return torch.exp(x)


class ParetoEps(EpsDistribution):
    """The configuration by Doksum 1987, see also example 4.7.1 in Bickel, Klassen, Ritov and Wellner
    P(\epsilon > t) = (1 + \eta e^t)^{-1/\eta}
    """

    def __init__(self, eta=1.):  # By default the proportional odds model
        self.eta = eta

    def hazard(self, x):
        return torch.exp(x) / (1 + self.eta * torch.exp(x))

    def log_hazard(self, x):
        return x - torch.log(1 + self.eta * torch.exp(x))

    def cumulative_hazard(self, x):
        return torch.log(1 + self.eta * torch.exp(x))


class NonparametricEps(EpsDistribution, nn.Module):  # This turns out to fail
    """Almost nonparametric version of a distribution with its cumulative hazard function approximated using
    a monotone neural net, the same idea in the paper https://arxiv.org/abs/1905.09690,
    However, it seems that brute force parameterization like this results in unbounded likelihood"""

    def __init__(self, num_hidden_units):
        nn.Module.__init__(self)
        self.ch = MonotoneMLP(num_hidden_units=num_hidden_units)

    def cumulative_hazard(self, x):
        return self.ch(x)

    def hazard(self, x):
        return self.ch.get_derivative(x)
