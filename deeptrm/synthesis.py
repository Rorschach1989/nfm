import torch
import torch.utils.data as data
import torch.nn.functional as F


__all__ = ('IdentityH', 'ExponentialH', 'M', 'Z', 'Noise', 'SyntheticData', 'PFSyntheticData')


class GenericH(object):
    """Configuration for oracle H, used in proportional frailty models"""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def reverse_call(self, x: torch.Tensor) -> torch.Tensor:
        """h.reverse_call(h(x)) = x"""
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError


class IdentityH(GenericH):

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def reverse_call(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def derivative(self, x):
        return torch.ones_like(x)


class ExponentialH(GenericH):

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x) - 1

    def reverse_call(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x)

    def derivative(self, x):
        return torch.exp(x)


class Z(object):

    def __init__(self, dist='uniform'):
        self.dist = dist

    def __call__(self, shape):
        if self.dist == 'uniform':
            return torch.rand(shape, requires_grad=False)
        else:
            raise NotImplementedError


class M(object):

    def __init__(self, z_law, params):
        self.z_law = z_law
        self.params = params

    def __call__(self, z):
        if self.z_law == 'linear':
            m_z = z @ self.params
        elif self.z_law == 'nonlinear_I':
            m_z = torch.sin(z) @ self.params + torch.sin(z @ self.params)
        elif self.z_law == 'nonlinear_II':
            d = z.shape[1]
            for dd in range(d):
                z[:, dd] = torch.pow(z[:, dd], dd+1)
            m_z = z @ self.params
        else:
            raise NotImplementedError
        return m_z


class Noise(object):

    def __init__(self, eps_dist, **kwargs):
        self.eps_dist = eps_dist
        self.kwargs = kwargs

    def __call__(self, shape):
        if self.eps_dist == 'gaussian':
            eps = torch.distributions.Normal(0., 1.).sample(shape)
        elif self.eps_dist == 'cox':
            eps = torch.log(- torch.log(1 - torch.rand(shape)))
        elif self.eps_dist == 'pareto':
            eta = self.kwargs.pop('eta', 1.)
            eps = torch.log((1 / torch.pow(1 - torch.rand(shape), eta) - 1) / eta)
        else:
            raise NotImplementedError
        return eps


class PFSyntheticData(data.Dataset):
    """Representing a survival style synthetic data, making things modular"""

    def __init__(self,
                 sample_size,
                 d,
                 censor_rate,
                 z_gen: Z,
                 H: GenericH,
                 m: M,
                 eps: Noise,
                 stochastic=True):
        self.sample_size = sample_size
        self.d = d
        self.censor_rate = censor_rate
        self.z_gen = z_gen
        self.H = H
        self.m = m
        self.eps = eps
        self.z = z_gen([sample_size, d])
        m_z = self.m(self.z)
        eps_event = self.eps(m_z.shape)
        eps_censor = self.eps(m_z.shape)
        self.t = self.H.reverse_call(-m_z + eps_event)
        self.c = self.H.reverse_call(-m_z + eps_censor + torch.rand(m_z.shape) - censor_rate)
        self.y = torch.minimum(self.t, self.c)
        if stochastic:
            self.y = self.y.clone().detach().requires_grad_(True)
        self.delta = (self.t <= self.c).type(torch.int)
        self.effective_sample_size = self.delta.sum()

    def sort(self):
        order = torch.argsort(self.y, dim=0)[:, 0]
        sort_y = self.y[order]
        sort_delta = self.delta[order]
        sort_z = self.z[order]
        return sort_y, sort_delta, sort_z

    def __getitem__(self, item):
        return self.z[item], self.y[item], self.delta[item]

    def __len__(self):
        return self.z.shape[0]

    # Utilities for oracle evaluations of distribution characteristics
    def oracle_hazard(self, z, y):
        h = self.H.derivative(y).view(-1)
        e_m_z = torch.exp(self.m(z).view(-1))
        return h * e_m_z


SyntheticData = PFSyntheticData
