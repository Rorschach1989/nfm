import numpy as np
import torch
import torchtuples as tt
from pycox.evaluation.eval_surv import EvalSurv
from pycox.models import CoxPH
from tqdm import tqdm

from deeptrm.datasets import SurvivalDataset


def np_convert(y_, delta_, z_):
    return y_.cpu().detach().numpy().reshape(-1, ), delta_.cpu().numpy().reshape(-1, ), z_.cpu().numpy()


# torch.manual_seed(77)
early_stopping_patience = 50
kkbox_train = SurvivalDataset.kkbox('train')
kkbox_valid = SurvivalDataset.kkbox('val')
kkbox_test = SurvivalDataset.kkbox('test')
fold_c_indices = []
fold_ibs = []
fold_inbll = []

np.random.seed(77)

for j in tqdm(range(5)):
    torch.manual_seed(77 + j)
    # Performance seems to be highly dependent on initialization, doing merely a 5-fold CV does NOT
    # seem to provide stable results, therefore repeat 10 times with distinct shuffle
    y, delta, z = np_convert(*kkbox_train.sort())
    y_valid, delta_valid, z_valid = np_convert(*kkbox_valid.sort())
    y_test, delta_test, z_test = np_convert(*kkbox_test.sort())

    valid = tt.tuplefy(z_valid, (y_valid, delta_valid))

    in_features = z.shape[1]
    num_nodes = [32, 32]
    out_features = 1
    batch_norm = True
    dropout = 0.1
    output_bias = False
    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                                  dropout, output_bias=output_bias)

    model = CoxPH(net, tt.optim.Adam)

    batch_size = 256
    model.optimizer.set_lr(0.01)

    epochs = 512
    callbacks = [tt.callbacks.EarlyStopping()]
    verbose = True

    log = model.fit(z, (y, delta), batch_size, epochs, callbacks, verbose,
                    val_data=valid)

    _ = model.compute_baseline_hazards()

    surv = model.predict_surv_df(z_test)

    ev = EvalSurv(surv, y_test, delta_test, censor_surv='km')
    time_grid = np.linspace(y_test.min(), y_test.max(), 100)
    # ev.concordance_td('antolini'), ev.integrated_brier_score(time_grid), ev.integrated_nbll(time_grid)

    fold_c_indices.append(ev.concordance_td('antolini'))
    fold_ibs.append(ev.integrated_brier_score(time_grid))
    fold_inbll.append(ev.integrated_nbll(time_grid))

print(
    np.around(np.asarray(fold_c_indices).mean(), 4),
    np.around(np.asarray(fold_ibs).mean(), 4),
    np.around(np.asarray(fold_inbll).mean(), 4)
)
print(
    np.around(np.asarray(fold_c_indices).std(), 4),
    np.around(np.asarray(fold_ibs).std(), 4),
    np.around(np.asarray(fold_inbll).std(), 4)
)

# [0.8386081136327392, 0.838711877250543, 0.8390307697528466, 0.8404354670931167, 0.8385520037596135]
# [0.11241340526206701, 0.11392272893327998, 0.11312548134691525, 0.11256387487241562, 0.11329878602278358]
# [0.3510578281305717, 0.3552122717216418, 0.35211198184494025, 0.3519252085824575, 0.35376948448796913]
