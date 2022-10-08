import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from nfm.datasets import SurvivalDataset
from nfm.base import FullyNeuralNLL
from nfm.eps_config import ParetoEps
from pycox.evaluation.eval_surv import EvalSurv


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=1 + data_full.num_features, out_features=128, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1, bias=False)
        )

    def forward(self, y, z):
        inputs = torch.cat([z, y], dim=1)
        return torch.exp(self.mlp(inputs))


data_full = SurvivalDataset.metabric('release/data/metabric_IHC4_clinical_train_test.h5')
fold_c_indices = []
fold_ibs = []
fold_nbll = []
normalizing_factor = 366.25


def normalize(y):
    return (y + 1) / normalizing_factor


for j in tqdm(range(10)):
    torch.manual_seed(77+j)
    train_folds, valid_folds, test_folds = data_full.cv_split(shuffle=True)
    for i in range(5):
        test_c_indices, test_ibs, test_nbll = [], [], []
        valid_losses = []
        nll = FullyNeuralNLL(eps_conf=ParetoEps(learnable=True), encoder=Net())
        optimizer = torch.optim.Adam(lr=1e-2, weight_decay=1e-3, params=nll.parameters())
        loader = DataLoader(train_folds[i], batch_size=128)
        for epoch in range(50):
            for z, y, delta in loader:
                nll.train()
                loss = nll(z=z, y=normalize(y), delta=delta)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            nll.eval()
            with torch.no_grad():
                y_valid, delta_valid, z_valid = valid_folds[i].sort()
                y_test, delta_test, z_test = test_folds[i].sort()
                y_valid, y_test = normalize(y_valid), normalize(y_test)
                valid_loss = nll(z_valid, y_valid, delta_valid)
                valid_losses.append(valid_loss)
                tg_test = np.linspace(y_test.cpu().numpy().min(), y_test.cpu().numpy().max(), 100)
                surv_pred_test = nll.get_survival_prediction(
                    z_test=z_test, y_test=torch.tensor(tg_test, dtype=torch.float).view(-1, 1))
                test_evaluator = EvalSurv(
                    surv=pd.DataFrame(surv_pred_test.cpu().numpy(), index=tg_test.reshape(-1)),
                    durations=y_test.cpu().numpy().reshape(-1),
                    events=delta_test.cpu().numpy().reshape(-1),
                    censor_surv='km')
                test_c_indices.append(test_evaluator.concordance_td(method='antolini'))
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
