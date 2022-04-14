import torch
import torch.nn as nn
import torch.nn.functional as F
from .monotone import MonotoneMLP
from .umnn import UMNN
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


class BoxCoxEps(EpsDistribution, nn.Module):
    """Box cox family of epsilon"""

    def __init__(self, eta=1., learnable=False):
        super(BoxCoxEps, self).__init__()
        if not learnable:
            self.log_eta = torch.log(torch.as_tensor(eta))
        else:
            self.log_eta = nn.Parameter(torch.log(torch.as_tensor(eta)), requires_grad=True)

    @property
    def eta(self):
        return torch.exp(self.log_eta)

    def hazard(self, x):
        return torch.pow(1 + torch.exp(x), self.eta - 1) * torch.exp(x)

    def log_hazard(self, x):
        return x + (self.eta - 1) * F.softplus(x)

    def cumulative_hazard(self, x):
        return (torch.pow(1 + torch.exp(x), self.eta) - 1) / self.eta


class PositiveStableEps(EpsDistribution, nn.Module):
    """The positive stable family"""

    def __init__(self, mu=1., learnable=False):
        super(PositiveStableEps, self).__init__()
        if not learnable:
            self.log_mu = torch.log(torch.as_tensor(mu))
        else:
            self.log_mu = nn.Parameter(torch.log(torch.as_tensor(mu)), requires_grad=True)

    @property
    def mu(self):
        return torch.exp(self.log_mu)

    def hazard(self, x):
        return self.mu * torch.exp(self.mu * x)

    def log_hazard(self, x):
        return self.log_mu + self.mu * x

    def cumulative_hazard(self, x):
        return torch.exp(self.mu * x)


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


class IGGEps(EpsDistribution, nn.Module):
    """The IGG family in the paper by Kosorok et al.,
    ROBUST INFERENCE FOR UNIVARIATE PROPORTIONAL HAZARDS FRAILTY REGRESSION MODELS
    """

    def __init__(self, alpha=0.5, gamma=1., alpha_learnable=False, gamma_learnable=True):
        super(IGGEps, self).__init__()
        logit_alpha = torch.log(torch.as_tensor(alpha) / (1 - torch.as_tensor(alpha)))
        if alpha_learnable:
            self.logit_alpha = nn.Parameter(logit_alpha, requires_grad=True)
        else:
            self.logit_alpha = logit_alpha
        if gamma_learnable:
            self.log_gamma = nn.Parameter(torch.log(torch.as_tensor(gamma)), requires_grad=True)
        else:
            self.log_gamma = torch.log(torch.as_tensor(gamma))

    def _major_term(self, x):
        return 1 + torch.exp(x + self.log_gamma) / (1 - self.alpha)

    @property
    def alpha(self):
        return torch.sigmoid(self.logit_alpha)

    @property
    def gamma(self):
        return torch.exp(self.log_gamma)

    def hazard(self, x):
        return torch.pow(self._major_term(x), self.alpha - 1) * torch.exp(x)

    def cumulative_hazard(self, x):
        return (torch.pow(self._major_term(x), self.alpha) - 1) * (1 - self.alpha) / (self.gamma * self.alpha)

    def log_hazard(self, x):
        return x + (self.alpha - 1) * torch.log(self._major_term(x))


class NonparametricEps(EpsDistribution, nn.Module):  # This turns out to fail
    """Almost nonparametric version of a distribution with its cumulative hazard function approximated using
    a monotone neural net, the same idea in the paper https://arxiv.org/abs/1905.09690,
    However, it seems that brute force parameterization like this results in unbounded likelihood"""

    def __init__(self, num_hidden_units):
        nn.Module.__init__(self)
        self.ch = UMNN(num_hidden_units=num_hidden_units)

    def cumulative_hazard(self, x):
        return self.ch(x)

    def hazard(self, x):
        return self.ch.get_derivative(x)
