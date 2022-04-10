import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from deeptrm.utils import default_device
from deeptrm.datasets import SurvivalDataset
from deeptrm.base import TransNLL, MonotoneNLL
from deeptrm.eps_config import GaussianEps, CoxEps, ParetoEps, NonparametricEps, BoxCoxEps, PositiveStableEps
from pycox.evaluation.eval_surv import EvalSurv


normalizing_factor = 1e3


def normalize(y):
    return (y + 1.) / normalizing_factor


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


rep_c_index, rep_ibs, rep_ibnll = [], [], []


for replicate in range(1):  # 10 for calculate std/mean
    m = nn.Sequential(
        nn.Linear(in_features=58, out_features=256, bias=False),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=1, bias=False)
    ).to(default_device)

    nll = MonotoneNLL(eps_conf=ParetoEps(learnable=True), num_hidden_units=256)
    optimizer = torch.optim.Adam(lr=1e-3, params=list(m.parameters()) + list(nll.parameters()))
    loader = DataLoader(kkbox_train, batch_size=256)
    for epoch in range(1):
        for i, (z, y, delta) in tqdm(enumerate(loader)):
            m.train()
            m_z = m(z)
            loss = nll(m_z=m_z, y=normalize(y), delta=delta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if not (i + 1) % 100:
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
                    valid_c_indices.append(valid_evaluator.concordance_td(method='antolini'))
                    test_c_indices.append(test_evaluator.concordance_td(method='antolini'))
                    valid_ibs.append(valid_evaluator.integrated_brier_score(time_grid=tg_valid))
                    test_ibs.append(test_evaluator.integrated_brier_score(time_grid=tg_test))
                    valid_inbll.append(valid_evaluator.integrated_nbll(time_grid=tg_valid))
                    test_inbll.append(test_evaluator.integrated_nbll(time_grid=tg_test))
                    print(loss,
                          valid_c_indices[-1],
                          test_c_indices[-1],
                          valid_ibs[-1],
                          test_ibs[-1],
                          valid_inbll[-1],
                          test_inbll[-1])


    valid_c_argmax = np.argmax(valid_c_indices)
    valid_ibs_argmin = np.argmin(valid_ibs)
    valid_inbll_argmin = np.argmin(valid_inbll)

