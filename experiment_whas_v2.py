import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from deeptrm.datasets import SurvivalDataset
from deeptrm.base import TransNLL, MonotoneNLL
from deeptrm.eps_config import GaussianEps, CoxEps, ParetoEps, NonparametricEps, BoxCoxEps, PositiveStableEps
from deeptrm.metric import c_index
from pycox.evaluation.eval_surv import EvalSurv


data_full = SurvivalDataset.whas('./data/whasncc.dat')
fold_c_indices = []
fold_ibs = []
fold_nbll = []
normalizing_factor = 1e3


def normalize(y):
    return (y + 1.) / normalizing_factor


for i in tqdm(range(10)):
    torch.manual_seed(77+i)
    train_folds, valid_folds, test_folds = data_full.cv_split(shuffle=True)
    for i in range(5):
        valid_c_indices, test_c_indices = [], []
        valid_ibs, test_ibs = [], []
        valid_nbll, test_nbll = [], []
        m = nn.Sequential(
            nn.Linear(in_features=8, out_features=512, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1, bias=False),
        )
        nll = MonotoneNLL(eps_conf=ParetoEps(learnable=True), num_hidden_units=256)
        optimizer = torch.optim.Adam(lr=1e-2, params=list(m.parameters()) + list(nll.parameters()))
        loader = DataLoader(train_folds[i], batch_size=128)
        for epoch in range(150):
            for z, y, delta in loader:
                m.train()
                m_z = m(z)
                loss = nll(m_z=m_z, y=normalize(y), delta=delta)
                # loss += 1e-3 * sum(p.pow(2.0).sum() for p in m.parameters())
                # loss += 1e-3 * sum(p.pow(2.0).sum() for p in nll.parameters())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            m.eval()
            with torch.no_grad():
                y_valid, delta_valid, z_valid = valid_folds[i].sort()
                y_test, delta_test, z_test = test_folds[i].sort()
                y_valid, y_test = normalize(y_valid), normalize(y_test)
                pred_valid = m(z_valid)
                pred_test = m(z_test)
                tg_valid = np.linspace(y_valid.numpy().min(), y_valid.numpy().max(), 100)
                tg_test = np.linspace(y_test.numpy().min(), y_test.numpy().max(), 100)
                surv_pred_valid = nll.get_survival_prediction(
                    pred_valid, y_test=torch.tensor(tg_valid, dtype=torch.float).view(-1, 1))
                surv_pred_test = nll.get_survival_prediction(
                    pred_test, y_test=torch.tensor(tg_test, dtype=torch.float).view(-1, 1))
                valid_evaluator = EvalSurv(
                    surv=pd.DataFrame(surv_pred_valid.numpy(), index=tg_valid.reshape(-1)),
                    durations=y_valid.numpy().reshape(-1),
                    events=delta_valid.numpy().reshape(-1),
                    censor_surv='km')
                test_evaluator = EvalSurv(
                    surv=pd.DataFrame(surv_pred_test.numpy(), index=tg_test.reshape(-1)),
                    durations=y_test.numpy().reshape(-1),
                    events=delta_test.numpy().reshape(-1),
                    censor_surv='km')
                valid_c_indices.append(valid_evaluator.concordance_td(method='antolini'))
                test_c_indices.append(test_evaluator.concordance_td(method='antolini'))
                valid_ibs.append(valid_evaluator.integrated_brier_score(time_grid=tg_valid))
                test_ibs.append(test_evaluator.integrated_brier_score(time_grid=tg_test))
                valid_nbll.append(valid_evaluator.integrated_nbll(time_grid=tg_valid))
                test_nbll.append(test_evaluator.integrated_nbll(time_grid=tg_test))
            valid_c_argmax = np.argmax(valid_c_indices)
            valid_ibs_argmin = np.argmin(valid_ibs)
            valid_nbll_argmin = np.argmin(valid_nbll)
            fold_c_indices.append(np.asarray(test_c_indices)[valid_c_argmax])
            fold_ibs.append(np.asarray(test_ibs)[valid_ibs_argmin])
            fold_nbll.append(np.asarray(test_nbll)[valid_nbll_argmin])


report_str = f"""
Results:
    mean c-index: {np.asarray(fold_c_indices).mean()}
    std c-index: {np.asarray(fold_c_indices).std()}
    mean ibs: {np.asarray(fold_ibs).mean()}
    std ibs: {np.asarray(fold_ibs).std()}
    mean ibll: {np.asarray(fold_nbll).mean()}
    std ibll: {np.asarray(fold_nbll).std()}
"""
print(report_str)
