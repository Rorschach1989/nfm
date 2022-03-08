import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from deeptrm.base import TransNLL
from deeptrm.eps_config import GaussianEps, CoxEps
from deeptrm.synthesis import SyntheticData

# torch.manual_seed(77)

sample_size = 1000
d = 5
censor_rate = 0.1


data = SyntheticData(sample_size, d, censor_rate, z_law='nonlinear_I', eps_dist='gaussian')
m = nn.Sequential(
    nn.Linear(in_features=5, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=1)
)
nll = TransNLL(eps_conf=GaussianEps(), num_jumps=data.effective_sample_size)
optimizer = torch.optim.Adam(lr=1e-1, params=list(m.parameters()) + list(nll.parameters()))

y, delta, z = data.sort()

for i in range(1000):
    m_z = m(z)
    loss = nll(m_z=m_z, y=y, delta=delta)
    optimizer.zero_grad()
    loss.backward()
    if not i % 100:
        print(loss)
    optimizer.step()

test_data = torch.rand([100, 5])
m_true = data.m_oracle(test_data).numpy().reshape(-1)
with torch.no_grad():
    m_est = m(test_data).numpy().reshape(-1)


fig, ax = plt.subplots(figsize=[20, 10])
ax.plot(m_true)
ax.plot(m_est)
plt.show()





