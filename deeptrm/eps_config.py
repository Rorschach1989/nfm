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


class ParetoEps(EpsDistribution, nn.Module):
    """The configuration by Doksum 1987, see also example 4.7.1 in Bickel, Klassen, Ritov and Wellner
    P(\epsilon > t) = (1 + \eta e^t)^{-1/\eta}
    """

    def __init__(self, eta=1., learnable=False):  # By default the proportional odds model
        nn.Module.__init__(self)
        if not learnable:
            self.log_eta = torch.log(torch.as_tensor(eta))
        else:
            # Essentially Gamma frailty model
            self.log_eta = nn.Parameter(torch.log(torch.as_tensor(eta)), requires_grad=True)

    @property
    def eta(self):
        return torch.exp(self.log_eta)

    def hazard(self, x):
        return torch.exp(x) / (1 + self.eta * torch.exp(x))

    def log_hazard(self, x):
        return x - torch.log(1 + self.eta * torch.exp(x))

    def cumulative_hazard(self, x):
        return torch.log(1 + self.eta * torch.exp(x)) / self.eta


class GaussianMixtureEps(GaussianEps, nn.Module):
    """A learnable gaussian mixture with pre-specified number of components"""

    def __init__(self, n_components):
        super(GaussianMixtureEps, self).__init__()
        nn.Module.__init__(self)
        self.n_components = n_components
        self.log_scale = nn.Parameter(torch.log(torch.rand([n_components])), requires_grad=True)
        self.mixture = nn.Parameter(torch.rand([n_components, 1]), requires_grad=True)

    def hazard(self, x):
        x = torch.exp(self.log_scale) * x.tile([1, self.n_components])
        component_hs = super(GaussianMixtureEps, self).hazard(x)
        return component_hs @ torch.softmax(self.mixture, dim=0)

    def cumulative_hazard(self, x):
        x = torch.exp(self.log_scale) * x.tile([1, self.n_components])
        component_hs = super(GaussianMixtureEps, self).cumulative_hazard(x)
        return component_hs @ torch.softmax(self.mixture, dim=0)

    def log_hazard(self, x):
        return torch.log(self.hazard(x))


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
