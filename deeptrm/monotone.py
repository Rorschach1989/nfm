import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from typing import Union
from .umnn import PositiveELU


class MonotoneLinear(nn.Module):
    """A slight modification of linear layer with positive constraint over weights
    via reparameterization, impl c.f. nn.Linear"""

    def __init__(self, in_features, out_features, weight_transform='abs'):
        super(MonotoneLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.weight_transform = getattr(torch, weight_transform)
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.weight)
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, 0.)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight_transform(self.weight), self.bias)


class SkipWrapper(nn.Module):
    """Wrapping a linear layer and activation/skip connections"""

    def __init__(self, linear: Union[nn.Linear, MonotoneLinear], act='tanh', skip_connection=True):
        super(SkipWrapper, self).__init__()
        self.linear = linear
        self.projection = MonotoneLinear(linear.in_features, linear.out_features)
        self.act = getattr(F, act, F.tanh)
        self.skip_connection = skip_connection

    def forward(self, x):
        nonlinear_out = self.act(self.linear(x))
        proj = self.projection(x)
        return nonlinear_out + proj if self.skip_connection else nonlinear_out


class MonotoneMLP(nn.Module):
    """A reference two-layer architecture for approximating UNIVARIATE monotone functions"""

    def __init__(self, num_hidden_units):
        super(MonotoneMLP, self).__init__()
        self._mlp = nn.Sequential(
            MonotoneLinear(in_features=1, out_features=num_hidden_units),
            nn.Tanh(),  # Sigmoid appears ok, ReLU is kinda weird
            MonotoneLinear(in_features=num_hidden_units, out_features=1),
            nn.Softplus()
        )

    def forward(self, x):
        return self._mlp(x)

    def get_derivative(self, x):
        y = self._mlp(x)
        y_prime, = torch.autograd.grad(y.sum(), x, create_graph=True)
        return y_prime
