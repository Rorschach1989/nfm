import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from nfm.base import TransNLL, MonotoneNLL
from nfm.eps_config import ParetoEps
from nfm.synthesis import *


torch.manual_seed(88888888)


sample_size = 5000
d = 5
censor_rate = 0.1
z_gen = Z(dist='uniform')
H = ExponentialH()
m_oracle = M(z_law='nonlinear_I', params=torch.arange(d).reshape([-1, 1]) / 10.)
eps = Noise(eps_dist='pareto', eta=1.)


# Three different scales of data, we illustrate only one of them, the rest is similar
data_small = SyntheticData(1000, d, censor_rate, z_gen, H, m_oracle, eps)
data_mid = SyntheticData(5000, d, censor_rate, z_gen, H, m_oracle, eps)
data_large = SyntheticData(10000, d, censor_rate, z_gen, H, m_oracle, eps)

# Test points
z_test = data_small.z_gen([100, d])
y_test = data_small.H.reverse_call(-m_oracle(z_test) + eps([100, 1]))

loader = DataLoader(data_small, batch_size=128)
m_ests, h_ests, nu_ests = [], [], []
for rep in tqdm(range(1)):
    m = nn.Sequential(
        nn.Linear(in_features=5, out_features=64, bias=False),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=1, bias=False)
    )
    nll = MonotoneNLL(eps_conf=ParetoEps(), num_hidden_units=64)
    optimizer = torch.optim.Adam(lr=1e-3, params=list(m.parameters()) + list(nll.parameters()), amsgrad=True)
    for epoch in range(100):
        for z, y, delta in loader:
            m_z = m(z)
            loss = nll(m_z=m_z, y=y, delta=delta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        m_est = m(z_test).view(-1).numpy()
        h_est = nll.h.get_derivative(y_test).view(-1).numpy()
        m_ests.append(m_est)
        h_ests.append(h_est)
        nu_ests.append(h_est * np.exp(m_est))


with torch.no_grad():
    m_true = m_oracle(z_test).numpy()
    m_est_ave = np.stack(m_ests, axis=1).mean(axis=1)

# Graphical illustration of m function estimation
fig, ax = plt.subplots(figsize=[20, 12])
ax.set_ylim([-.75, 2.5])
ax.plot(m_true, '-o', label='ground truth')
ax.plot(m_est_ave, '--', color='red', alpha=0.7, label='estimated')
ax.set_xlabel('index')
ax.set_ylabel('m(Z)')
ax.legend(loc='upper left', fontsize=20)
ax.set_title(f'N={sample_size}')
plt.show()


with torch.no_grad():
    nu_true = np.log(data_small.oracle_hazard(z_test, y_test).view(-1).numpy())
    nu_est_ave = np.log(np.stack(nu_ests, axis=1).mean(axis=1))

# Graphical illustration of nu function estimation
fig, ax = plt.subplots(figsize=[20, 12])
ax.set_ylim([0, 5])
ax.plot(nu_true, '-o', label='ground truth')
ax.plot(nu_est_ave, '--', color='red', alpha=0.7, label='estimated')
ax.legend()
ax.set_title(f'N={sample_size}')
plt.show()