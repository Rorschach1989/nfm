import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pycox.evaluation.eval_surv import EvalSurv
from torch.utils.data import DataLoader
from tqdm import tqdm

from deeptrm.datasets import SurvivalDataset
from deeptrm.utils import default_device
from sumo.loss import SuMoLoss

kkbox_train = SurvivalDataset.kkbox('train')
kkbox_valid = SurvivalDataset.kkbox('val')
kkbox_test = SurvivalDataset.kkbox('test')
fold_c_indices = []
fold_ibs = []
fold_nbll = []
normalizing_factor = 366.25


def normalize(t):
    return (t + 1) / normalizing_factor


n_hidden = 128


for j in tqdm(range(1)):
    torch.manual_seed(77 + j)

    test_c_indices, test_ibs, test_nbll = [], [], []
    m = nn.Sequential(
        nn.Linear(in_features=kkbox_train.z.shape[-1], out_features=n_hidden),
        nn.Tanh(),
        # nn.Dropout(),
        nn.Linear(in_features=n_hidden, out_features=n_hidden),
        # nn.Dropout()
    ).to(default_device)
    # nll = MonotoneNLL(eps_conf=ParetoEps(learnable=True), num_hidden_units=256)
    nll = SuMoLoss(in_features=n_hidden, num_hidden_units=n_hidden, weight_transform='square')
    optimizer = torch.optim.Adam(lr=1e-2, weight_decay=1e-2, params=list(m.parameters()) + list(nll.parameters()))
    loader = DataLoader(kkbox_train, batch_size=128)

    num_epoch = 1
    min_valid_loss = np.inf
    patience = 5
    trigger = 0

    for epoch in range(num_epoch):
        for z, y, delta in tqdm(loader):
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
        y_valid, delta_valid, z_valid = kkbox_valid.sort()
        y_test, delta_test, z_test = kkbox_test.sort()
        y_valid, y_test = normalize(y_valid), normalize(y_test)
        y_valid = y_valid.clone().requires_grad_(True)
        pred_valid = m(z_valid)
        valid_loss = nll(pred_valid, y_valid, delta_valid)
        with torch.no_grad():
            if (valid_loss.item() < min_valid_loss) or (epoch == num_epoch - 1):
                torch.save(m, 'model.pth')
                min_valid_loss = valid_loss.item()
                trigger = 0
            else:
                trigger += 1
            if (trigger >= patience) or (epoch == num_epoch - 1):
                m = torch.load('model.pth')
                pred_test = m(z_test)

                tg_test = np.linspace(y_test.cpu().numpy().min(), y_test.cpu().numpy().max(), 100)
                surv_pred_test = nll.get_survival_prediction(
                    pred_test, y_test=torch.tensor(tg_test, dtype=torch.float).view(-1, 1).to(default_device),
                    batch=True)
                test_evaluator = EvalSurv(
                    surv=pd.DataFrame(surv_pred_test.cpu().numpy(), index=tg_test.reshape(-1)),
                    durations=y_test.cpu().numpy().reshape(-1),
                    events=delta_test.cpu().numpy().reshape(-1),
                    censor_surv='km')
                fold_c_indices.append(test_evaluator.concordance_td(method='antolini'))
                fold_ibs.append(test_evaluator.integrated_brier_score(time_grid=tg_test))
                fold_nbll.append(test_evaluator.integrated_nbll(time_grid=tg_test))
                break

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
