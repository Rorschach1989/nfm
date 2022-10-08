import torch
import torch.nn as nn
from nfm.monotone import MonotoneLinear


class SuMoLoss(nn.Module):
    """A sleeker implementing the SuMo loss function in the paper
    `Survival Regression with Proper Scoring Rules and Monotonic Neural Networks`
    """

    def __init__(self, in_features, num_hidden_units, weight_transform='abs'):  # two-layer architecture
        super(SuMoLoss, self).__init__()
        self.num_hidden_units = num_hidden_units
        self.h_mixed = nn.Sequential(
            MonotoneLinear(in_features=in_features + 1,
                           out_features=num_hidden_units,
                           weight_transform=weight_transform),
            nn.Tanh(),
            MonotoneLinear(num_hidden_units, 1, weight_transform='abs')
        )

    def forward(self, m_z, y, delta):
        inputs = torch.cat([m_z, y], dim=1)
        h = self.h_mixed(inputs)
        # Compute survival
        s = 1 - torch.sigmoid(h)
        neg_f, = torch.autograd.grad(s.sum(), y, create_graph=True)
        f = -neg_f
        return -(delta * torch.log(f + 1e-15) + (1 - delta) * torch.log(s + 1e-15)).mean()

    def get_survival_prediction(self, m_z, y_test, batch=False, size=None):
        batch_size = m_z.shape[0]
        n_times = y_test.shape[0]
        if batch:
            if size is None:
                size = m_z.shape[0] // m_z.shape[1]
            temp = list()
            for i in range(batch_size // size + 1):
                m_z_ = torch.tile(m_z[i * size:(i + 1) * size], [n_times]).view(-1, n_times, m_z.shape[1])
                y_test_ = torch.tile(y_test, [m_z_.shape[0], 1]).view(-1, n_times, 1)
                inputs = torch.cat([m_z_, y_test_], dim=-1)
                temp.append(1 - torch.sigmoid(self.h_mixed(inputs).view(m_z_.shape[0], n_times)))
            s = torch.concat(temp)
        else:
            m_z_ = torch.tile(m_z, [n_times]).view(batch_size, n_times, -1)
            y_test_ = torch.tile(y_test, [batch_size, 1]).view(batch_size, n_times, 1)
            inputs = torch.cat([m_z_, y_test_], dim=-1)
            s = 1 - torch.sigmoid(self.h_mixed(inputs).view(batch_size, n_times))
        return s.T
