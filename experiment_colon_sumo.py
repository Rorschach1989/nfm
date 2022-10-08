import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from nfm.datasets import SurvivalDataset
from nfm.base import TransNLL, MonotoneNLL
from sumo.loss import SuMoLoss
from nfm.eps_config import GaussianEps, CoxEps, ParetoEps, NonparametricEps, BoxCoxEps, PositiveStableEps
from nfm.metric import c_index
from pycox.evaluation.eval_surv import EvalSurv


data_full = SurvivalDataset.colon('./data/colon.csv')
# data_full.apply_scaler(standardize=False)
fold_c_indices = []
fold_ibs = []
fold_nbll = []
normalizing_factor = 366.25


def normalize(y):
    return (y + 1) / normalizing_factor
    # return y


for i in tqdm(range(10)):
    torch.manual_seed(77+i)
    train_folds, valid_folds, test_folds = data_full.cv_split(shuffle=True)
    for i in range(5):
        test_c_indices, test_ibs, test_nbll = [], [], []
        valid_losses = []
        m = nn.Sequential(
            nn.Linear(in_features=25, out_features=128),
            nn.Tanh(),
            # nn.Dropout(),
            nn.Linear(in_features=128, out_features=128),
            # nn.Dropout()
        )
        # nll = MonotoneNLL(eps_conf=ParetoEps(learnable=True), num_hidden_units=256)
        nll = SuMoLoss(in_features=128, num_hidden_units=128)
        optimizer = torch.optim.Adam(lr=1e-3, weight_decay=1e-3, params=list(m.parameters()) + list(nll.parameters()))
        loader = DataLoader(train_folds[i], batch_size=128)
        for epoch in range(50):
            for z, y, delta in loader:
                m.train()
                m_z = m(z)
                y = y.clone().requires_grad_(True)
                loss = nll(m_z=m_z, y=normalize(y), delta=delta)
                # loss += 1e-3 * sum(p.pow(2.0).sum() for p in m.parameters())
                # loss += 1e-3 * sum(p.pow(2.0).sum() for p in nll.parameters())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            m.eval()
            y_valid, delta_valid, z_valid = valid_folds[i].sort()
            y_test, delta_test, z_test = test_folds[i].sort()
            y_valid, y_test = normalize(y_valid), normalize(y_test)
            y_valid = y_valid.clone().requires_grad_(True)
            pred_valid = m(z_valid)
            pred_test = m(z_test)
            valid_loss = nll(pred_valid, y_valid, delta_valid)
            with torch.no_grad():
                valid_losses.append(valid_loss.numpy())
                tg_test = np.linspace(y_test.cpu().numpy().min(), y_test.cpu().numpy().max(), 100)
                surv_pred_test = nll.get_survival_prediction(
                    pred_test, y_test=torch.tensor(tg_test, dtype=torch.float).view(-1, 1))
                test_evaluator = EvalSurv(
                    surv=pd.DataFrame(surv_pred_test.cpu().numpy(), index=tg_test.reshape(-1)),
                    durations=y_test.cpu().numpy().reshape(-1),
                    events=delta_test.cpu().numpy().reshape(-1),
                    censor_surv='km')
                # valid_c_indices.append(valid_evaluator.concordance_td(method='antolini'))
                test_c_indices.append(test_evaluator.concordance_td(method='antolini'))
                # test_c_indices.append(c_index(-pred_test, y_test, delta_test))
                test_ibs.append(test_evaluator.integrated_brier_score(time_grid=tg_test))
                test_nbll.append(test_evaluator.integrated_nbll(time_grid=tg_test))
        valid_argmin = np.argmin(valid_losses)
        fold_c_indices.append(np.asarray(test_c_indices)[valid_argmin])
        fold_ibs.append(np.asarray(test_ibs)[valid_argmin])
        fold_nbll.append(np.asarray(test_nbll)[valid_argmin])


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
