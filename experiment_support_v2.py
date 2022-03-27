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

torch.manual_seed(77)
data_full = SurvivalDataset.support('./data/support_train_test.h5')
fold_c_indices = []
fold_ibs = []
normalizing_factor = 1e3


for _ in tqdm(range(1)):
    train_folds, valid_folds, test_folds = data_full.cv_split(shuffle=True)
    for i in range(5):
        valid_c_indices, test_c_indices = [], []
        valid_ibs, test_ibs = [], []
        m = nn.Sequential(
            nn.Linear(in_features=14, out_features=128, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1, bias=False)
        )
        nll = MonotoneNLL(eps_conf=ParetoEps(learnable=True), num_hidden_units=256)
        optimizer = torch.optim.Adam(lr=1e-3, params=list(m.parameters()) + list(nll.parameters()))
        loader = DataLoader(train_folds[i], batch_size=128)
        for epoch in range(50):
            for z, y, delta in loader:
                m.train()
                m_z = m(z)
                loss = nll(m_z=m_z, y=y/normalizing_factor, delta=delta)
                # loss += 1e-3 * sum(p.pow(2.0).sum() for p in m.parameters())
                # loss += 1e-3 * sum(p.pow(2.0).sum() for p in nll.parameters())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            m.eval()
            with torch.no_grad():
                y_valid, delta_valid, z_valid = valid_folds[i].sort()
                y_test, delta_test, z_test = test_folds[i].sort()
                y_valid, y_test = y_valid/normalizing_factor, y_test/normalizing_factor
                pred_valid = m(z_valid)
                pred_test = m(z_test)
                tg_valid = np.linspace(y_valid.numpy().min(), y_valid.numpy().max(), 100)
                tg_test = np.linspace(y_test.numpy().min(), y_test.numpy().max(), 100)
                surv_pred_valid = nll.get_survival_prediction(pred_valid, y_test=y_valid)
                surv_pred_test = nll.get_survival_prediction(pred_test, y_test=y_test)
                valid_evaluator = EvalSurv(
                    surv=pd.DataFrame(surv_pred_valid.numpy(), index=y_valid.numpy().reshape(-1)),
                    durations=y_valid.numpy().reshape(-1),
                    events=delta_valid.numpy().reshape(-1),
                    censor_surv='km')
                test_evaluator = EvalSurv(
                    surv=pd.DataFrame(surv_pred_test.numpy(), index=y_test.numpy().reshape(-1)),
                    durations=y_test.numpy().reshape(-1),
                    events=delta_test.numpy().reshape(-1),
                    censor_surv='km')
                valid_c_indices.append(valid_evaluator.concordance_td(method='antolini'))
                test_c_indices.append(test_evaluator.concordance_td(method='antolini'))
                valid_ibs.append(valid_evaluator.integrated_brier_score(time_grid=tg_valid))
                test_ibs.append(test_evaluator.integrated_brier_score(time_grid=tg_test))
            valid_c_argmax = np.argmax(valid_c_indices)
            valid_ibs_argmin = np.argmin(valid_ibs)
            # print(valid_argmax)
            fold_c_indices.append(np.asarray(test_c_indices)[valid_c_argmax])
            fold_ibs.append(np.asarray(test_ibs)[valid_ibs_argmin])

print(np.asarray(fold_c_indices).mean(), np.asarray(fold_ibs).mean())
