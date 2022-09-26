import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from deeptrm.utils import default_device
from deeptrm.datasets import SurvivalDataset
from deeptrm.base import TransNLL, MonotoneNLL
from deeptrm.eps_config import GaussianEps, CoxEps, ParetoEps, NonparametricEps, BoxCoxEps, PositiveStableEps, IGGEps
from pycox.evaluation.eval_surv import EvalSurv
from itertools import product

normalizing_factor = 366.25


def normalize(t):
    return (t + 1.) / normalizing_factor


kkbox_train = SurvivalDataset.kkbox('train')
kkbox_valid = SurvivalDataset.kkbox('val')
kkbox_test = SurvivalDataset.kkbox('test')
valid_c_indices, test_c_indices = [], []
valid_ibs, test_ibs = [], []
valid_inbll, test_inbll = [], []
y_valid, delta_valid, z_valid = kkbox_valid.sort()
y_test, delta_test, z_test = kkbox_test.sort()
y_valid = normalize(y_valid)
y_test = normalize(y_test)

np.random.seed(0)
idx_valid = torch.from_numpy(np.random.choice(len(kkbox_valid), 10 ** 4, replace=False)).long()
idx_test = torch.from_numpy(np.random.choice(len(kkbox_test), 10 ** 4, replace=False)).long()
y_valid, delta_valid, z_valid = y_valid[idx_valid], delta_valid[idx_valid], z_valid[idx_valid]
y_test, delta_test, z_test = y_test[idx_test], delta_test[idx_test], z_test[idx_test]

rep_c_index, rep_ibs, rep_ibnll = [], [], []


hyper_params = tuple(product(
    (16, 32, 64, 128),
    (
        IGGEps(alpha_learnable=True),
        CoxEps(),
        ParetoEps(learnable=True),
        BoxCoxEps(learnable=True),
        PositiveStableEps(learnable=True)
    ),
    (256, 512),
    (1e-3, 1e-2),
    (1e-3, 1e-2),
))
hp_indices = np.random.choice(len(hyper_params), 120, replace=False)
for hp_idx in hp_indices:
    hidden_size, eps_dist, batch_size, lr, weight_decay = hyper_params[hp_idx]
    print(f'hidden_size: {hidden_size}, eps_dist: {type(eps_dist).__name__}, batch_size: {batch_size}')
    for replicate in range(1):  # 10 for calculate std/mean
        m = nn.Sequential(
            nn.Linear(in_features=58, out_features=hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=1, bias=False)
        ).to(default_device)

        nll = MonotoneNLL(eps_conf=eps_dist, num_hidden_units=256)
        optimizer = torch.optim.Adam(lr=lr, params=list(m.parameters()) + list(nll.parameters()), weight_decay=weight_decay)
        loader = DataLoader(kkbox_train, batch_size=batch_size)
        for epoch in range(1):
            for i, (z, y, delta) in enumerate(loader):
                m.train()
                m_z = m(z)
                loss = nll(m_z=m_z, y=normalize(y), delta=delta)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            m.eval()
            with torch.no_grad():
                pred_valid = m(z_valid)
                pred_test = m(z_test)
                tg_valid = np.linspace(
                    y_valid.cpu().numpy().min(), y_valid.cpu().numpy().max(), 100).astype(np.float32)
                tg_test = np.linspace(
                    y_test.cpu().numpy().min(), y_test.cpu().numpy().max(), 100).astype(np.float32)
                surv_pred_valid = nll.get_survival_prediction(  # place into GPU device
                    pred_valid, y_test=torch.tensor(tg_valid).to(default_device).view(-1, 1))
                surv_pred_test = nll.get_survival_prediction(
                    pred_test, y_test=torch.tensor(tg_test).to(default_device).view(-1, 1))
                valid_evaluator = EvalSurv(
                    surv=pd.DataFrame(surv_pred_valid.cpu().numpy(), index=tg_valid.reshape(-1)),
                    durations=y_valid.cpu().numpy().reshape(-1),
                    events=delta_valid.cpu().numpy().reshape(-1),
                    censor_surv='km')
                test_evaluator = EvalSurv(
                    surv=pd.DataFrame(surv_pred_test.cpu().numpy(), index=tg_test.reshape(-1)),
                    durations=y_test.cpu().numpy().reshape(-1),
                    events=delta_test.cpu().numpy().reshape(-1),
                    censor_surv='km')
                valid_c_indices.append(valid_evaluator.concordance_td())
                test_c_indices.append(test_evaluator.concordance_td())
                valid_ibs.append(valid_evaluator.integrated_brier_score(time_grid=tg_valid))
                test_ibs.append(test_evaluator.integrated_brier_score(time_grid=tg_test))
                valid_inbll.append(valid_evaluator.integrated_nbll(time_grid=tg_valid))
                test_inbll.append(test_evaluator.integrated_nbll(time_grid=tg_test))
                print(f'{valid_c_indices[-1]:.6f}, {valid_ibs[-1]:.6f}, {valid_inbll[-1]:.6f}, {test_c_indices[-1]:.6f}, {test_ibs[-1]:.6f}, {test_inbll[-1]:.6f}')

valid_c_argmax = np.argmax(valid_c_indices)
valid_ibs_argmin = np.argmin(valid_ibs)
valid_inbll_argmin = np.argmin(valid_inbll)
print(f'bst:\n{valid_c_argmax}, {valid_ibs_argmin}, {valid_inbll_argmin}')

# 256
# 0.84306762888664 0.8429995595158624 0.11189790569055066 0.1115977328015469 0.35699031867630904 0.3556520979705805
# 512
# 0.8439156406463844 0.8439519333314783 0.11071846737653786 0.11037853381376403 0.35513775689786603 0.3537193201930532
