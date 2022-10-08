import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from nfm.datasets import SurvivalDataset
from nfm.base import TransNLL, MonotoneNLL
from nfm.eps_config import GaussianEps, CoxEps, ParetoEps, NonparametricEps, PositiveStableEps, BoxCoxEps
from nfm.metric import c_index
from pycox.evaluation.eval_surv import EvalSurv

pretrain = False
# torch.manual_seed(77)
early_stopping_patience = 50
data_full = SurvivalDataset.metabric('./data/metabric_IHC4_clinical_train_test.h5')
data_full.apply_scaler(standardize=False)
fold_c_indices = []
fold_ibs = []
fold_inbll = []


n_hidden = 256


for i in tqdm(range(10)):
    torch.manual_seed(77 + i)
    # Performance seems to be highly dependent on initialization, doing merely a 5-fold CV does NOT
    # seem to provide stable results, therefore repeat 10 times with distinct shuffle
    train_folds, valid_folds, test_folds = data_full.cv_split(shuffle=True)
    for i in range(5):
        y, delta, z = train_folds[i].sort()
        y_valid, delta_valid, z_valid = valid_folds[i].sort()
        y_test, delta_test, z_test = test_folds[i].sort()
        y = y.clone().detach().requires_grad_(False)
        test_c_indices, test_ibs, test_nbll = [], [], []
        valid_c_indices = []
        c = nn.Sequential(
            nn.Linear(in_features=13, out_features=n_hidden, bias=False),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=n_hidden, out_features=1, bias=False),
        )
        umnn_nll = MonotoneNLL(eps_conf=ParetoEps(learnable=True), num_hidden_units=256)
        umnn_optimizer = torch.optim.Adam(
            lr=1e-3, weight_decay=1e-2, params=list(c.parameters()) + list(umnn_nll.parameters()))
        loader = DataLoader(train_folds[i], batch_size=128)
        val_losses = []
        min_loss = np.inf
        non_improvement = 0
        for epoch in range(50):
            for z_, y_, delta_ in loader:
                c.train()
                c_z = c(z_)
                loss = umnn_nll(m_z=c_z, y=y_, delta=delta_)
                umnn_optimizer.zero_grad()
                loss.backward()
                umnn_optimizer.step()
                c.eval()
            with torch.no_grad():
                val_loss = umnn_nll(c(z_valid), y_valid, delta_valid).cpu().numpy()
                if val_loss > min_loss:
                    non_improvement += 1
                else:
                    min_loss = val_loss
                if non_improvement >= 5:
                    break

        if pretrain:
            m = c
        else:
            m = nn.Sequential(
                nn.Linear(in_features=13, out_features=n_hidden, bias=False),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=n_hidden, out_features=1, bias=False),
            )
        nll = TransNLL(eps_conf=ParetoEps(learnable=True), num_jumps=int(train_folds[i].delta.sum()))
        with torch.no_grad():
            h = umnn_nll.h.get_derivative(c(z)).masked_select(delta.type(torch.bool))
            nll.log_jump_sizes.copy_(torch.log(h + 1e-15).requires_grad_(True))
        optimizer = torch.optim.Adam(
            params=[
                {'params': m.parameters(), 'lr': 1e-3, 'weight_decay': 1e-2},
                {'params': nll.parameters(), 'lr': 1e-4}
            ]
        )
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
                    pred_valid = m(z_valid)
                    pred_test = m(z_test)
                    tg_valid = np.linspace(y_valid.numpy().min(), y_valid.numpy().max(), 100)
                    tg_test = np.linspace(y_test.numpy().min(), y_test.numpy().max(), 100)
                    surv_pred_valid = nll.get_survival_prediction(pred_valid, y_test=torch.tensor(tg_valid).view(-1, 1))
                    surv_pred_test = nll.get_survival_prediction(pred_test, y_test=torch.tensor(tg_test).view(-1, 1))
                    test_evaluator = EvalSurv(
                        surv=pd.DataFrame(surv_pred_test.numpy(), index=tg_test.reshape(-1)),
                        durations=y_test.numpy().reshape(-1),
                        events=delta_test.numpy().reshape(-1),
                        censor_surv='km')
                    valid_c_indices.append(c_index(-pred_valid, y_valid, delta_valid))
                    test_c_indices.append(c_index(-pred_test, y_test, delta_test))
                    test_ibs.append(test_evaluator.integrated_brier_score(time_grid=tg_test))
                    test_nbll.append(test_evaluator.integrated_nbll(time_grid=tg_test))
        valid_c_argmax = np.argmax(valid_c_indices)
        fold_c_indices.append(np.asarray(test_c_indices)[valid_c_argmax])
        fold_ibs.append(np.asarray(test_ibs)[valid_c_argmax])
        fold_inbll.append(np.asarray(test_nbll)[valid_c_argmax])

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