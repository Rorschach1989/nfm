import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from nfm.base import TransNLL
from nfm.eps_config import GaussianEps, CoxEps, ParetoEps
from nfm.synthesis import SyntheticData

# torch.manual_seed(77)

sample_size = 10000
d = 5
censor_rate = 0.1


data = SyntheticData(sample_size, d, censor_rate, z_law='nonlinear_I', eps_dist='pareto')
m = nn.Sequential(
    nn.Linear(in_features=5, out_features=128, bias=False),  # TODO: Including bias may result in weird behavior
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=1, bias=False)
)
nll = TransNLL(eps_conf=ParetoEps(), num_jumps=data.effective_sample_size)
optimizer = torch.optim.Adam(lr=1e-1, params=list(m.parameters()) + list(nll.parameters()))

y, delta, z = data.sort()
print(delta.sum())
test_data = torch.rand([100, 5])
m_true = data.m_oracle(test_data).numpy().reshape(-1)

fig, ax = plt.subplots(figsize=[20, 12])
ax.plot(m_true, '-o', label='ground truth')

for i in range(5000):
    m_z = m(z)
    loss = nll(m_z=m_z, y=y, delta=delta)
    optimizer.zero_grad()
    loss.backward()
    if not i % 1000:
        print(loss)
        with torch.no_grad():
            m_est = m(test_data).numpy().reshape(-1)
        ax.plot(m_est, label=f'iteration {i}')
    optimizer.step()

ax.legend()
ax.set_title(f'N={sample_size}')
plt.show()