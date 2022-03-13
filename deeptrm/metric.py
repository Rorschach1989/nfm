import torch


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
