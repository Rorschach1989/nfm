# Implemntations adapted from UMNN's official libray at
# https://github.com/AWehenkel/UMNN
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .utils import default_device


def _flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


def compute_cc_weights(nb_steps):
    lam = np.arange(0, nb_steps + 1, 1).reshape(-1, 1)
    lam = np.cos((lam @ lam.T) * math.pi / nb_steps)
    lam[:, 0] = .5
    lam[:, -1] = .5 * lam[:, -1]
    lam = lam * 2 / nb_steps
    W = np.arange(0, nb_steps + 1, 1).reshape(-1, 1)
    W[np.arange(1, nb_steps + 1, 2)] = 0
    W = 2 / (1 - W ** 2)
    W[0] = 1
    W[np.arange(1, nb_steps + 1, 2)] = 0
    cc_weights = torch.tensor(lam.T @ W).float()
    steps = torch.tensor(np.cos(np.arange(0, nb_steps + 1, 1).reshape(-1, 1) * math.pi / nb_steps)).float()

    return cc_weights, steps


def integrate(x0, nb_steps, step_sizes, integrand, compute_grad=False, x_tot=None):
    #Clenshaw-Curtis Quadrature Method
    cc_weights, steps = compute_cc_weights(nb_steps)

    device = x0.get_device() if x0.is_cuda else "cpu"
    cc_weights, steps = cc_weights.to(device), steps.to(device)

    xT = x0 + nb_steps*step_sizes
    if not compute_grad:
        x0_t = x0.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        xT_t = xT.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        steps_t = steps.unsqueeze(0).expand(x0_t.shape[0], -1, x0_t.shape[2])
        X_steps = x0_t + (xT_t-x0_t)*(steps_t + 1)/2
        X_steps = X_steps.contiguous().view(-1, x0_t.shape[2])
        dzs = integrand(X_steps)
        dzs = dzs.view(xT_t.shape[0], nb_steps+1, -1)
        dzs = dzs*cc_weights.unsqueeze(0).expand(dzs.shape)
        z_est = dzs.sum(1)
        return z_est*(xT - x0)/2
    else:
        x0_t = x0.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        xT_t = xT.unsqueeze(1).expand(-1, nb_steps + 1, -1)
        x_tot = x_tot * (xT - x0) / 2
        x_tot_steps = x_tot.unsqueeze(1).expand(-1, nb_steps + 1, -1) \
                      * cc_weights.unsqueeze(0).expand(x_tot.shape[0], -1, x_tot.shape[1])
        steps_t = steps.unsqueeze(0).expand(x0_t.shape[0], -1, x0_t.shape[2])
        X_steps = x0_t + (xT_t - x0_t) * (steps_t + 1) / 2
        X_steps = X_steps.contiguous().view(-1, x0_t.shape[2])
        x_tot_steps = x_tot_steps.contiguous().view(-1, x_tot.shape[1])

        g_param = computeIntegrand(X_steps, integrand, x_tot_steps, nb_steps+1)
        return g_param


def computeIntegrand(x, integrand, x_tot, nb_steps):
    with torch.enable_grad():
        f = integrand(x)
        g_param = _flatten(torch.autograd.grad(f, integrand.parameters(), x_tot, create_graph=True, retain_graph=True))
    return g_param


class ParallelNeuralIntegral(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x0, x, integrand, flat_params, nb_steps=20):
        with torch.no_grad():
            x_tot = integrate(x0, nb_steps, (x - x0)/nb_steps, integrand, False)
            # Save for backward
            ctx.integrand = integrand
            ctx.nb_steps = nb_steps
            ctx.save_for_backward(x0.clone(), x.clone())
        return x_tot

    @staticmethod
    def backward(ctx, grad_output):
        x0, x = ctx.saved_tensors
        integrand = ctx.integrand
        nb_steps = ctx.nb_steps
        integrand_grad = integrate(x0, nb_steps, x/nb_steps, integrand, True, grad_output)
        x_grad = integrand(x)
        x0_grad = integrand(x0)
        # Leibniz formula
        return -x0_grad*grad_output, x_grad*grad_output, None, integrand_grad, None


class PositiveELU(nn.Module):

    def __init__(self):
        super(PositiveELU, self).__init__()

    def forward(self, x):
        return F.elu(x) + 1


class UMNN(nn.Module):
    """The UMNN model in the paper
    `Unconstrained Monotonic Neural Networks`
    """

    def __init__(self, num_hidden_units, nb_steps=50, positive_transform='elu1p'):
        super(UMNN, self).__init__()
        positive_layer = PositiveELU() if positive_transform == 'elu1p' else nn.Softplus()
        self._derivative_mlp = nn.Sequential(
            nn.Linear(in_features=1, out_features=num_hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=num_hidden_units, out_features=1),
            positive_layer,
        )
        self.nb_steps = nb_steps

    def get_derivative(self, x):
        return self._derivative_mlp(x)

    def forward(self, x):
        shape = x.shape
        x = x.view(-1, 1)
        x0 = torch.zeros(x.shape).to(default_device)
        return ParallelNeuralIntegral.apply(
            x0, x, self._derivative_mlp, _flatten(self._derivative_mlp.parameters()), self.nb_steps).view(shape)
