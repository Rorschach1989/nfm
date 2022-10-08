import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from nfm.utils import default_device
from nfm.datasets import SurvivalDataset
from nfm.base import TransNLL, MonotoneNLL
from nfm.eps_config import GaussianEps, CoxEps, ParetoEps, NonparametricEps, BoxCoxEps, PositiveStableEps
from pycox.evaluation.eval_surv import EvalSurv
from sksurv.linear_model import CoxPHSurvivalAnalysis


mimic_train = SurvivalDataset.mimiciii('train')
mimic_valid = SurvivalDataset.mimiciii('valid')
mimic_test = SurvivalDataset.mimiciii('test')
valid_c_indices, test_c_indices = [], []
valid_ibs, test_ibs = [], []
valid_inbll, test_inbll = [], []
y_train, delta_train, z_train = mimic_train.sort()
y_valid, delta_valid, z_valid = mimic_valid.sort()
y_test, delta_test, z_test = mimic_test.sort()


with torch.no_grad():
    y_train = y_train.numpy()
    delta_train = delta_train.numpy()
    y_test = y_test.numpy()
    delta_test = delta_test.numpy()


train_tuple = np.array(
    [(bool(_d), _y) for _y, _d in zip(y_train.reshape(-1), delta_train.reshape(-1))],
    dtype=[('e.tdm', '?'), ('t.tdm', '<f8')]
)
rsf = CoxPHSurvivalAnalysis(alpha=1e-2)
rsf.fit(z_train, train_tuple)
survs = rsf.predict_survival_function(z_test)
y_min, y_max = y_train.min(), y_train.max()
tg_test = np.linspace(  # Overflow is not allowed in sksurv
    max(y_min, y_test.min()), min(y_max, y_test.max()), 100)
# tg_test = np.linspace(y_test.min(), y_test.max(), 100)
preds = np.asarray([[fn(t) for t in tg_test] for fn in tqdm(survs)])
test_evaluator = EvalSurv(
    surv=pd.DataFrame(preds.T, index=tg_test.reshape(-1)),
    durations=y_test.reshape(-1),
    events=delta_test.reshape(-1),
    censor_surv='km')
test_tuple = np.array(
    [(bool(_d), _y) for _y, _d in zip(np.clip(y_test, y_min, y_max).reshape(-1), delta_test.reshape(-1))],
    # [(bool(_d), _y) for _y, _d in zip(y_test.reshape(-1), delta_test.reshape(-1))],
    dtype=[('e.tdm', '?'), ('t.tdm', '<f8')]
)
print(test_evaluator.concordance_td(method='antolini'))
print(test_evaluator.integrated_brier_score(time_grid=tg_test))
print(test_evaluator.integrated_nbll(time_grid=tg_test))

