import torch
import numba
import numpy as np


def c_index(y_pred, y_true, delta):
    """implementation of the concordance index, essentially a censoring-compatible version of Kendall's tau

    Args:
        y_pred(torch.Tensor): predicted survival upto monotone transforms, with shape [batch_size, 1]
        y_true(torch.Tensor): true observed time, with shape [batch_size, 1]
        delta(torch.Tensor): observed delta, with shape [batch_size, 1]
    """
    m1 = ((y_pred - y_pred.view(1, -1)) < 0)
    m2 = ((y_true - y_true.view(1, -1)) < 0)
    c_pred = (m1 * m2 * delta).sum()
    c_true = (m2 * delta).sum()
    return c_pred / c_true


@numba.njit(parallel=True)
def c_index_large_scale(y_pred, y_true, delta):
    """Acceleration of conventional c-index via JIT

    Args:
        y_pred(np.ndarray): rank-1 array of shape [batch_size,]
        y_true(np.ndarray): rank-1 array of shape [batch_size,]
        delta(np.ndarray): rank-1 array of shape [batch_size,]
    """
    nom = 0.
    denom = 0.
    n = y_pred.shape[0]
    for i in numba.prange(n):
        for j in numba.prange(n):
            m1_ij = numba.float32(y_pred[i] < y_pred[j])
            m2_ij = numba.float32(y_true[i] < y_true[j])
            nom = nom + m1_ij * m2_ij * delta[i]
            denom = denom + m2_ij * delta[i]
    return nom / denom
