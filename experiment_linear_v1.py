import torch
import numpy as np
import torch.nn as nn
from nfm.base import TransNLL
from nfm.eps_config import GaussianEps, CoxEps
from nfm.synthesis import SyntheticData

# torch.manual_seed(77)

sample_size = 1000
d = 5
censor_rate = 0.1


def trial():
    data = SyntheticData(sample_size, d, censor_rate, eps_dist='gaussian')
    m = torch.nn.Linear(in_features=5, out_features=1, bias=False)
    nll = TransNLL(eps_conf=GaussianEps(), num_jumps=data.effective_sample_size)
    optimizer = torch.optim.Adam(lr=1e-1, params=list(m.parameters()) + list(nll.parameters()))
    y, delta, z = data.sort()

    for i in range(2000):
        m_z = m(z)
        loss = nll(m_z=m_z, y=y, delta=delta)
        optimizer.zero_grad()
        loss.backward()
        # if not i % 100:
        #     print(loss)
        optimizer.step()

    m.eval()
    return list(m.parameters())[0].detach().numpy()


results = []
for _ in range(100):
    results.append(trial())

results = np.concatenate(results, axis=0)
print(results.mean(axis=0))




