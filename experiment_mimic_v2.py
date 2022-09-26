import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from deeptrm.utils import default_device
from deeptrm.datasets import SurvivalDataset
from deeptrm.base import TransNLL, MonotoneNLL, FullyNeuralNLL
from deeptrm.eps_config import GaussianEps, CoxEps, ParetoEps, NonparametricEps, BoxCoxEps, PositiveStableEps
from pycox.evaluation.eval_surv import EvalSurv


torch.manual_seed(7777777)


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
        # self.te = TimeEncoder(d_time=32)
        self.mlp = nn.Sequential(
            # nn.Linear(in_features=64 + 26, out_features=128),
            nn.Linear(in_features=1 + 26, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1)
        )

    def forward(self, y, z):
        # inputs = torch.cat([z, self.te(y)], dim=1)
        inputs = torch.cat([z, y], dim=1)
        return torch.exp(self.mlp(inputs))


normalizing_factor = 1e7


def normalize(y):
    return (y + 1.) / normalizing_factor


mimic_train = SurvivalDataset.mimiciii('train')
mimic_valid = SurvivalDataset.mimiciii('valid')
mimic_test = SurvivalDataset.mimiciii('test')
y_valid, delta_valid, z_valid = mimic_valid.sort()
y_test, delta_test, z_test = mimic_test.sort(persist=True)
y_valid = normalize(y_valid)
y_test = normalize(y_test)


rep_c_index, rep_ibs, rep_ibnll = [], [], []


for replicate in range(10):  # 10 for calculate std/mean
    valid_losses = []
    # m = nn.Sequential(
    #     nn.Linear(in_features=25, out_features=32, bias=False),
    #     nn.ReLU(),
    #     nn.Linear(in_features=32, out_features=1, bias=False)
    # )
    # nll = MonotoneNLL(eps_conf=ParetoEps(learnable=True),
    #                   num_hidden_units=128,
    #                   positive_transform='elu1p')
    nll = FullyNeuralNLL(eps_conf=ParetoEps(learnable=True), encoder=Net())
    optimizer = torch.optim.Adam(lr=1e-2, weight_decay=1e-3, params=nll.parameters())
    test_c_indices, test_ibs, test_nbll = [], [], []
    loader = DataLoader(mimic_train, batch_size=256)
    test_loader = DataLoader(mimic_test, batch_size=1024)
    for epoch in tqdm(range(50)):
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
            valid_loss = nll(z_valid, y_valid, delta_valid)
            valid_losses.append(valid_loss)
            tg_test = np.linspace(y_test.cpu().numpy().min(), y_test.cpu().numpy().max(), 100)
            # surv_pred_test = nll.get_survival_prediction(
            #     z_test=z_test, y_test=torch.tensor(tg_test, dtype=torch.float).view(-1, 1))
            surv_pred_test = []
            for z_t, _, _ in test_loader:
                elem = nll.get_survival_prediction(
                    z_test=z_t, y_test=torch.tensor(tg_test, dtype=torch.float).view(-1, 1))
                surv_pred_test.append(elem)
            surv_pred_test = torch.cat(surv_pred_test, dim=1)
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


