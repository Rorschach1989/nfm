import torch
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
