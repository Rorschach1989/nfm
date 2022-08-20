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


torch.manual_seed(7777777)


normalizing_factor = 1e8


def normalize(y):
    return (y + 1.) / normalizing_factor


mimic_train = SurvivalDataset.mimiciii('train')
mimic_valid = SurvivalDataset.mimiciii('valid')
mimic_test = SurvivalDataset.mimiciii('test')
y_valid, delta_valid, z_valid = mimic_valid.sort()
y_test, delta_test, z_test = mimic_test.sort()
y_valid = normalize(y_valid)
y_test = normalize(y_test)


rep_c_index, rep_ibs, rep_ibnll = [], [], []


for replicate in range(1):  # 10 for calculate std/mean
    m = nn.Sequential(
        nn.Linear(in_features=26, out_features=256, bias=False),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=1, bias=False)
    ).to(default_device)
    valid_losses = []
    test_c_indices, test_ibs, test_nbll = [], [], []
    nll = MonotoneNLL(eps_conf=ParetoEps(learnable=True), num_hidden_units=256)
    optimizer = torch.optim.Adam(lr=1e-2, params=list(m.parameters()) + list(nll.parameters()))
    loader = DataLoader(mimic_train, batch_size=256)
    for epoch in tqdm(range(50)):
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
            valid_loss = nll(pred_valid, y_valid, delta_valid)
            valid_losses.append(valid_loss.cpu().numpy())
            tg_test = np.linspace(y_test.cpu().numpy().min(), y_test.cpu().numpy().max(), 100)
            surv_pred_test = nll.get_survival_prediction(
                pred_test, y_test=torch.tensor(tg_test, dtype=torch.float).view(-1, 1).to(default_device))
            test_evaluator = EvalSurv(
                surv=pd.DataFrame(surv_pred_test.cpu().numpy(), index=tg_test.reshape(-1)),
                durations=y_test.cpu().numpy().reshape(-1),
                events=delta_test.cpu().numpy().reshape(-1),
                censor_surv='km')
            # valid_c_indices.append(valid_evaluator.concordance_td(method='antolini'))
            # test_c_indices.append(test_evaluator.concordance_td(method='antolini'))
            test_c_indices.append(test_evaluator.concordance_td(method='antolini'))
            test_ibs.append(test_evaluator.integrated_brier_score(time_grid=tg_test))
            test_nbll.append(test_evaluator.integrated_nbll(time_grid=tg_test))
    valid_argmin = np.argmin(valid_losses)
    rep_c_index.append(np.asarray(test_c_indices)[valid_argmin])
    rep_ibs.append(np.asarray(test_ibs)[valid_argmin])
    rep_ibnll.append(np.asarray(test_nbll)[valid_argmin])


report_str = f"""
Results:
    mean c-index: {np.asarray(rep_c_index).mean()}
    std c-index: {np.asarray(rep_c_index).std()}
    mean ibs: {np.asarray(rep_ibs).mean()}
    std ibs: {np.asarray(rep_ibs).std()}
    mean ibll: {np.asarray(rep_ibnll).mean()}
    std ibll: {np.asarray(rep_ibnll).std()}
"""
print(report_str)


