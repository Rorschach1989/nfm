import torch
import torch.utils.data as data


class SyntheticData(data.Dataset):
    """Representing a survival style synthetic data"""

    def __init__(self,
                 sample_size,
                 d,
                 censor_rate,
                 z_law='linear',
                 h_inv_true='identity',
                 eps_dist='gaussian'):
        self.sample_size = sample_size
        self.d = d
        self.censor_rate = censor_rate
        self.z = torch.rand([sample_size, d], requires_grad=False)
        self.z_law = z_law
        m_z = self.m_oracle(self.z)
        if eps_dist == 'gaussian':
            eps = torch.distributions.Normal(0., 1.).sample(m_z.shape)
        elif eps_dist == 'cox':
            eps = torch.log(- torch.log(1 - torch.rand(m_z.shape)))
        else:
            raise NotImplementedError
        self.t = -m_z + eps
        self.c = -m_z + torch.rand(m_z.shape) - censor_rate  # Approx
        self.y = torch.minimum(self.t, self.c)
        self.delta = (self.t <= self.c).type(torch.int)
        self.effective_sample_size = self.delta.sum()

    def m_oracle(self, z):
        if self.z_law == 'linear':
            self.z_params = torch.arange(self.d).reshape([-1, 1]) / 10.
            m_z = z @ self.z_params
        elif self.z_law == 'nonlinear_I':
            self.z_params = torch.arange(self.d).reshape([-1, 1]) / 10.
            m_z = torch.sin(z) @ self.z_params + torch.sin(z @ self.z_params)
        else:
            raise NotImplementedError
        return m_z

    def sort(self):
        order = torch.argsort(self.y, dim=0)[:, 0]
        sort_y = self.y[order]
        sort_delta = self.delta[order]
        sort_z = self.z[order]
        return sort_y, sort_delta, sort_z

    def __getitem__(self, item):
        return self.z[item], self.y[item], self.delta[item]

