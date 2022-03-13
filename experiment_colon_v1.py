import torch
import torch.nn as nn
from tqdm import tqdm
from deeptrm.datasets import Colon
from deeptrm.base import TransNLL, MonotoneNLL
from deeptrm.eps_config import GaussianEps, CoxEps, ParetoEps, NonparametricEps
from deeptrm.metric import c_index

torch.manual_seed(77)
data_full = Colon.from_csv('./data/colon.csv')
fold_c_indices = []

for _ in tqdm(range(10)):
    # Performance seems to be highly dependent on initialization, doing merely a 5-fold CV does NOT
    # seem to provide stable results, therefore repeat 10 times with distinct shuffle
    train_folds, valid_folds, test_folds = data_full.cv_split(shuffle=True)
    for i in range(5):
        valid_c_indices, test_c_indices = [], []
        m = nn.Sequential(
            nn.Linear(in_features=25, out_features=128, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1, bias=False),
        )
        nll = TransNLL(eps_conf=GaussianEps(), num_jumps=int(train_folds[i].delta.sum()))
        optimizer = torch.optim.Adam(lr=1e-3, weight_decay=1e-3, params=list(m.parameters()) + list(nll.parameters()))
        y, delta, z = train_folds[i].sort()
        for j in range(1000):
            m.train()
            m_z = m(z)
            loss = nll(m_z=m_z, y=y, delta=delta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            m.eval()
            with torch.no_grad():
                pred_valid = -m(valid_folds[i].z)
                pred_test = -m(test_folds[i].z)
                c_valid = c_index(y_pred=pred_valid, y_true=valid_folds[i].y, delta=valid_folds[i].delta)
                c_test = c_index(y_pred=pred_test, y_true=test_folds[i].y, delta=test_folds[i].delta)
                valid_c_indices.append(c_valid)
                test_c_indices.append(c_test)
        valid_argmax = torch.argmax(torch.tensor(valid_c_indices))
        # print(valid_argmax)
        fold_c_indices.append(torch.tensor(test_c_indices)[valid_argmax])


print(f'Final c index {float(torch.tensor(fold_c_indices).mean())}')

