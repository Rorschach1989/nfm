import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from nfm.base import TransNLL, MonotoneNLL
from nfm.eps_config import GaussianEps, CoxEps, ParetoEps, NonparametricEps
from nfm.synthesis import SyntheticData

torch.manual_seed(77)

sample_size = 10000
d = 5
censor_rate = 0.1


data = SyntheticData(sample_size, d, censor_rate, z_law='nonlinear_I', eps_dist='pareto')
loader = DataLoader(data, batch_size=128)
m = nn.Sequential(
    nn.Linear(in_features=5, out_features=128, bias=False),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=1, bias=False)
)
nll = MonotoneNLL(eps_conf=ParetoEps(), num_hidden_units=128)
# Fact: SGD works fine, Adamax is somewhat OK, Adam is problematic
# TODO: resort to modifications of adam like the one by Satyen Kale
optimizer = torch.optim.SGD(lr=1e-2, momentum=0.9, params=list(m.parameters()) + list(nll.parameters()))
# optimizer = torch.optim.Adamax(lr=1e-2, params=list(m.parameters()) + list(nll.parameters()))

print(data.delta.sum())
test_data = torch.rand([100, 5])
m_true = data.m_oracle(test_data).numpy().reshape(-1)

fig, ax = plt.subplots(figsize=[20, 12])
ax.plot(m_true, '-o', label='ground truth')

i = 0
loss = None
for _ in range(100):
    for z, y, delta in loader:
        m_z = m(z)
        loss = nll(m_z=m_z, y=y, delta=delta)
        optimizer.zero_grad()
        loss.backward()
        i += 1
        if not i % 1000 and i > 5000:
            print(i, loss)
            with torch.no_grad():
                m_est = m(test_data).numpy().reshape(-1)
            ax.plot(m_est, label=f'iteration {i}')
        optimizer.step()

# Plot the last step
print(i, loss)
with torch.no_grad():
    m_est = m(test_data).numpy().reshape(-1)
ax.plot(m_est, label=f'iteration {i}')
ax.legend()
ax.set_title(f'N={sample_size}')
plt.show()