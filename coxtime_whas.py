import numpy as np
import torch
import torchtuples as tt
from pycox.evaluation.eval_surv import EvalSurv
from pycox.models import CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime
from tqdm import tqdm

from deeptrm.datasets import SurvivalDataset

torch.manual_seed(77)
early_stopping_patience = 50
data_full = SurvivalDataset.whas('./data/whasncc.dat')
fold_c_indices = []
fold_ibs = []
fold_inbll = []

np.random.seed(77)

for _ in tqdm(range(10)):
    # Performance seems to be highly dependent on initialization, doing merely a 5-fold CV does NOT
    # seem to provide stable results, therefore repeat 10 times with distinct shuffle
    train_folds, valid_folds, test_folds = data_full.cv_split(shuffle=True)
    for i in range(5):
        def np_convert(y_, delta_, z_):
            return y_.detach().numpy().reshape(-1, ), delta_.numpy().reshape(-1, ), z_.numpy()


        y, delta, z = np_convert(*train_folds[i].sort())
        y_valid, delta_valid, z_valid = np_convert(*valid_folds[i].sort())
        y_test, delta_test, z_test = np_convert(*test_folds[i].sort())

        labtrans = CoxTime.label_transform()
        y, delta = labtrans.fit_transform(y, delta)
        y_valid, delta_valid = labtrans.transform(y_valid, delta_valid)
        valid = tt.tuplefy(z_valid, (y_valid, delta_valid))

        in_features = z.shape[1]
        num_nodes = [32, 32]
        batch_norm = True
        dropout = 0.1
        net = MLPVanillaCoxTime(in_features, num_nodes, batch_norm, dropout)

        model = CoxTime(net, tt.optim.Adam, labtrans=labtrans)

        batch_size = 256
        model.optimizer.set_lr(0.01)

        epochs = 512
        callbacks = [tt.callbacks.EarlyStopping()]
        verbose = True

        log = model.fit(z, (y, delta), batch_size, epochs, callbacks, verbose,
                        val_data=valid.repeat(10).cat())

        _ = model.compute_baseline_hazards()

        surv = model.predict_surv_df(z_test)

        ev = EvalSurv(surv, y_test, delta_test, censor_surv='km')
        time_grid = np.linspace(y_test.min(), y_test.max(), 100)
        ev.concordance_td(), ev.integrated_brier_score(time_grid), ev.integrated_nbll(time_grid)

        fold_c_indices.append(ev.concordance_td())
        fold_ibs.append(ev.integrated_brier_score(time_grid))
        fold_inbll.append(ev.integrated_nbll(time_grid))

print(
    np.around(np.asarray(fold_c_indices).mean(), 3),
    np.around(np.asarray(fold_ibs).mean(), 3),
    np.around(np.asarray(fold_inbll).mean(), 3)
)
print(
    np.around(np.asarray(fold_c_indices).std(), 3),
    np.around(np.asarray(fold_ibs).std(), 3),
    np.around(np.asarray(fold_inbll).std(), 3)
)
