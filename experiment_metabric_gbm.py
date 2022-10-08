import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from nfm.datasets import SurvivalDataset
from nfm.metric import c_index
from pycox.evaluation.eval_surv import EvalSurv
from sksurv.ensemble import GradientBoostingSurvivalAnalysis


data_full = SurvivalDataset.metabric('./data/metabric_IHC4_clinical_train_test.h5')

fold_c_indices = []
fold_ibs = []
fold_inbll = []


for i in tqdm(range(10)):
    torch.manual_seed(77 + i)
    train_folds, valid_folds, test_folds = data_full.cv_split(shuffle=True)
    for j in range(5):
        with torch.no_grad():
            y_train, delta_train, z_train = map(lambda y: y.numpy(), train_folds[j].sort())
            y_test, delta_test, z_test = map(lambda y: y.numpy(), test_folds[j].sort())
        train_tuple = np.array(
            [(bool(_d), _y) for _y, _d in zip(y_train.reshape(-1), delta_train.reshape(-1))],
            dtype=[('e.tdm', '?'), ('t.tdm', '<f8')]
        )
        est_cph_tree = GradientBoostingSurvivalAnalysis(
            n_estimators=200, learning_rate=0.05, max_depth=2, random_state=0
        )
        est_cph_tree.fit(z_train, train_tuple)
        survs = est_cph_tree.predict_survival_function(z_test)
        y_min, y_max = y_train.min(), y_train.max()
        tg_test = np.linspace(  # Overflow is not allowed in sksurv
            max(y_min, y_test.min()), min(y_max, y_test.max()), 100)
        preds = np.asarray([[fn(t) for t in tg_test] for fn in survs])
        test_evaluator = EvalSurv(
            surv=pd.DataFrame(preds.T, index=tg_test.reshape(-1)),
            durations=y_test.reshape(-1),
            events=delta_test.reshape(-1),
            censor_surv='km')
        fold_c_indices.append(test_evaluator.concordance_td())
        fold_ibs.append(test_evaluator.integrated_brier_score(time_grid=tg_test))
        fold_inbll.append(test_evaluator.integrated_nbll(time_grid=tg_test))


report_str = f"""
Results:
    mean c-index: {np.asarray(fold_c_indices).mean()}
    std c-index: {np.asarray(fold_c_indices).std()}
    mean ibs: {np.asarray(fold_ibs).mean()}
    std ibs: {np.asarray(fold_ibs).std()}
    mean ibll: {np.asarray(fold_inbll).mean()}
    std ibll: {np.asarray(fold_inbll).std()}
"""
print(report_str)

