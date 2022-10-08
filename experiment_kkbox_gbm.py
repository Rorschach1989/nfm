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
from xgbse import XGBSEKaplanNeighbors


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


xgb_params = {
    "objective": "survival:cox",
    "eval_metric": "cox-nloglik",
    "tree_method": "exact",
    "max_depth": 3,
    "booster": "gbtree",
    "subsample": 1.0,
    "min_child_weight": 30,
    "colsample_bynode": 1.0,
    'n_jobs': -1,
}


rep_c_index, rep_ibs, rep_ibnll = [], [], []


for i in tqdm(range(1)):
    torch.manual_seed(77 + i)
    for j in tqdm(range(5)):
        with torch.no_grad():
            y_train, delta_train, z_train = map(lambda y: y.numpy(), kkbox_train.sort())
            y_test, delta_test, z_test = map(lambda y: y.numpy(), [y_test, delta_test, z_test])
        train_tuple = np.array(
            [(bool(_d), _y) for _y, _d in zip(y_train.reshape(-1), delta_train.reshape(-1))],
            dtype=[('e.tdm', '?'), ('t.tdm', '<f8')]
        )
        est_cph_tree = XGBSEKaplanNeighbors(xgb_params=xgb_params)
        est_cph_tree.fit(z_train, train_tuple, num_boost_round=10, persist_train=False)
        y_min, y_max = y_train.min(), y_train.max()
        tg_test = np.linspace(  # Overflow is not allowed in sksurv
            max(y_min, y_test.min()), min(y_max, y_test.max()), 100)
        preds = est_cph_tree.predict(z_test, time_bins=tg_test)
        # preds = np.asarray([[fn(t) for t in tg_test] for fn in survs])
        test_evaluator = EvalSurv(
            surv=pd.DataFrame(preds.T, index=tg_test.reshape(-1)),
            durations=y_test.reshape(-1),
            events=delta_test.reshape(-1),
            censor_surv='km')
        rep_c_index.append(test_evaluator.concordance_td())
        rep_ibs.append(test_evaluator.integrated_brier_score(time_grid=tg_test))
        rep_ibnll.append(test_evaluator.integrated_nbll(time_grid=tg_test))

