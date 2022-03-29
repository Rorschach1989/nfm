import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from deeptrm.datasets import SurvivalDataset
from deeptrm.base import TransNLL, MonotoneNLL
from deeptrm.eps_config import GaussianEps, CoxEps, ParetoEps, NonparametricEps
from deeptrm.metric import c_index
from pycox.evaluation.eval_surv import EvalSurv

torch.manual_seed(77)
early_stopping_patience = 50
data_full = SurvivalDataset.support('./data/support_train_test.h5')
fold_c_indices = []
fold_ibs = []


for _ in tqdm(range(10)):
    # Performance seems to be highly dependent on initialization, doing merely a 5-fold CV does NOT
    # seem to provide stable results, therefore repeat 10 times with distinct shuffle
    train_folds, valid_folds, test_folds = data_full.cv_split(shuffle=True)
    for i in range(5):
        y, delta, z = train_folds[i].sort()
        y = y.clone().detach().requires_grad_(False)
        valid_c_indices, test_c_indices = [], []
        valid_ibs, test_ibs = [], []
        m = nn.Sequential(
            nn.Linear(in_features=14, out_features=256, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1, bias=True),
        )
        nll = TransNLL(eps_conf=ParetoEps(learnable=True), num_jumps=int(train_folds[i].delta.sum()))
        optimizer = torch.optim.Adam(lr=1e-3, params=list(m.parameters()) + list(nll.parameters()))
        for j in range(1000):
            m.train()
            m_z = m(z)
            loss = nll(m_z=m_z, y=y, delta=delta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            m.eval()
            if not (j + 1) % early_stopping_patience:
                with torch.no_grad():
                    y_valid, delta_valid, z_valid = valid_folds[i].sort()
                    y_test, delta_test, z_test = test_folds[i].sort()
                    pred_valid = m(z_valid)
                    pred_test = m(z_test)
                    tg_valid = np.linspace(y_valid.numpy().min(), y_valid.numpy().max(), 100)
                    tg_test = np.linspace(y_test.numpy().min(), y_test.numpy().max(), 100)
                    surv_pred_valid = nll.get_survival_prediction(pred_valid, y_test=torch.tensor(tg_valid).view(-1, 1))
                    surv_pred_test = nll.get_survival_prediction(pred_test, y_test=torch.tensor(tg_test).view(-1, 1))
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
                    valid_c_indices.append(valid_evaluator.concordance_td())
                    test_c_indices.append(test_evaluator.concordance_td())
                    valid_ibs.append(valid_evaluator.integrated_brier_score(time_grid=tg_valid))
                    test_ibs.append(test_evaluator.integrated_brier_score(time_grid=tg_test))
        valid_c_argmax = np.argmax(valid_c_indices)
        valid_ibs_argmin = np.argmin(valid_ibs)
        # print(valid_argmax)
        fold_c_indices.append(np.asarray(test_c_indices)[valid_c_argmax])
        fold_ibs.append(np.asarray(test_ibs)[valid_ibs_argmin])


print(np.asarray(fold_c_indices).mean(), np.asarray(fold_ibs).mean())
