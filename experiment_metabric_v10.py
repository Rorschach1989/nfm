import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from deeptrm.datasets import SurvivalDataset
from deeptrm.base import TransNLL, MonotoneNLL, FullyNeuralNLL
from deeptrm.monotone import SkipWrapper
from deeptrm.eps_config import GaussianEps, CoxEps, ParetoEps, NonparametricEps, BoxCoxEps, PositiveStableEps, IGGEps
from deeptrm.metric import c_index
from pycox.evaluation.eval_surv import EvalSurv


class TimeEncoder(nn.Module):

    def __init__(self, d_time=32):
        super(TimeEncoder, self).__init__()
        self.d_time = torch.tensor(d_time, dtype=torch.float, requires_grad=False)
        self.basis_freq = nn.Parameter(1 / torch.pow(10, torch.linspace(0, 1.5, d_time)))

    def forward(self, t):  # t is rank-1
        map_ts = t.view(-1, 1) * self.basis_freq.view(1, -1)
        harmonic = torch.cat([torch.cos(map_ts), torch.sin(map_ts)], dim=1)
        return harmonic / torch.sqrt(self.d_time)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.te = TimeEncoder()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=2 * 32 + 13, out_features=64, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1, bias=False)
        )

    def forward(self, y, z):
        inputs = torch.cat([z, self.te(y)], dim=1)
        return F.softplus(self.mlp(inputs))


class NetV2(nn.Module):
    """Test MHA using time as query"""

    def __init__(self):
        super(NetV2, self).__init__()
        self.te = TimeEncoder()
        self.mha = nn.MultiheadAttention(embed_dim=64, kdim=1, vdim=1, num_heads=1, batch_first=True)
        self.z_forward = nn.Linear(in_features=13, out_features=32)
        self.out_forward = nn.Linear(in_features=64, out_features=1)

    def forward(self, y, z):
        q = torch.unsqueeze(self.te(y), dim=1)  # [n, 1, 2 * d_time]
        k = v = torch.unsqueeze(self.z_forward(z), dim=-1)  # [n, num_hidden, 1]
        attn, _ = self.mha(q, k, v)  # [n, 1, embed_dim]
        return torch.exp(self.out_forward(torch.squeeze(attn, dim=1)))


data_full = SurvivalDataset.metabric('./data/metabric_IHC4_clinical_train_test.h5')
# data_full.apply_scaler(standardize=False)
fold_c_indices = []
fold_ibs = []
fold_nbll = []
normalizing_factor = 366.25


def normalize(y):
    return (y + 1) / normalizing_factor


for j in tqdm(range(1)):
    torch.manual_seed(77+j)
    train_folds, valid_folds, test_folds = data_full.cv_split(shuffle=True)
    for i in range(5):
        test_c_indices, test_ibs, test_nbll = [], [], []
        valid_losses = []
        # m = nn.Sequential(
        #     nn.Linear(in_features=25, out_features=32, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(in_features=32, out_features=1, bias=False)
        # )
        # nll = MonotoneNLL(eps_conf=ParetoEps(learnable=True),
        #                   num_hidden_units=128,
        #                   positive_transform='elu1p')
        nll = FullyNeuralNLL(eps_conf=ParetoEps(learnable=True), encoder=NetV2())
        optimizer = torch.optim.Adam(lr=1e-2, weight_decay=1e-3, params=nll.parameters())
        loader = DataLoader(train_folds[i], batch_size=128)
        for epoch in range(50):
            for z, y, delta in loader:
                nll.train()
                loss = nll(z=z, y=normalize(y), delta=delta)
                # loss += 1e-3 * sum(p.abs().sum() for p in m.parameters())
                # loss += 1e-3 * sum(p.abs().sum() for p in nll.parameters())
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
                # valid_c_indices.append(valid_evaluator.concordance_td(method='antolini'))
                test_c_indices.append(test_evaluator.concordance_td(method='antolini'))
                test_ibs.append(test_evaluator.integrated_brier_score(time_grid=tg_test))
                test_nbll.append(test_evaluator.integrated_nbll(time_grid=tg_test))
        valid_argmin = np.argmin(valid_losses)
        # print(valid_argmin)
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
