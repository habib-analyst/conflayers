"""Set-valued (conformal) layers.

Each layer maps Interval -> Interval. Affine layers (Linear, Conv2d,
BatchNorm at inference) use *exact* interval arithmetic. Monotone
elementwise layers (ReLU, MaxPool) use exact monotone propagation.

The LEARNED part: affine layers carry a per-output-unit slack parameter
s >= 0 (via softplus) that widens the interval. Slack is what training
optimizes: widen where the point map is unreliable, shrink where it is
tight — minimizing mean set size subject to empirical coverage.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .intervals import Interval, affine_interval, monotone_interval


class _SlackMixin:
    def _make_slack(self, n: int, init: float = -3.0):
        # softplus(-3) ~= 0.05: start nearly tight, let training widen.
        self.log_slack = nn.Parameter(torch.full((n,), init))

    @property
    def slack(self) -> torch.Tensor:
        return F.softplus(self.log_slack)

    def _widen(self, iv: Interval, slack: torch.Tensor) -> Interval:
        # Broadcast slack over all dims except the channel/unit dim.
        while slack.dim() < iv.lo.dim():
            slack = slack.unsqueeze(0)
        return Interval(iv.lo - slack, iv.hi + slack)


class ConformalLinear(nn.Module, _SlackMixin):
    """Set-valued linear layer: exact affine interval map + learned slack."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self._make_slack(out_features)

    def forward(self, z: Interval) -> Interval:
        out = affine_interval(self.weight, self.bias, z)
        return self._widen(out, self.slack)

    @classmethod
    def from_linear(cls, lin: nn.Linear) -> "ConformalLinear":
        c = cls(lin.in_features, lin.out_features, bias=lin.bias is not None)
        with torch.no_grad():
            c.weight.copy_(lin.weight)
            if lin.bias is not None:
                c.bias.copy_(lin.bias)
        return c


class ConformalConv2d(nn.Module, _SlackMixin):
    """Set-valued 2D convolution: exact interval arithmetic + learned slack."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=padding, bias=bias)
        self._make_slack(out_ch)

    def _conv_interval(self, z: Interval) -> Interval:
        W, b = self.conv.weight, self.conv.bias
        kw = dict(stride=self.conv.stride, padding=self.conv.padding,
                  dilation=self.conv.dilation, groups=self.conv.groups)
        Wpos = torch.clamp(W, min=0.0)
        Wneg = torch.clamp(W, max=0.0)
        lo = (F.conv2d(z.lo, Wpos, None, **kw) + F.conv2d(z.hi, Wneg, None, **kw))
        hi = (F.conv2d(z.hi, Wpos, None, **kw) + F.conv2d(z.lo, Wneg, None, **kw))
        if b is not None:
            lo = lo + b.view(1, -1, 1, 1)
            hi = hi + b.view(1, -1, 1, 1)
        return Interval(lo, hi)

    def forward(self, z: Interval) -> Interval:
        out = self._conv_interval(z)
        return self._widen(out, self.slack.view(1, -1, 1, 1))

    @classmethod
    def from_conv(cls, conv: nn.Conv2d) -> "ConformalConv2d":
        ks = conv.kernel_size[0] if isinstance(conv.kernel_size, tuple) else conv.kernel_size
        c = cls(conv.in_channels, conv.out_channels, ks,
                stride=conv.stride, padding=conv.padding, bias=conv.bias is not None)
        with torch.no_grad():
            c.conv.weight.copy_(conv.weight)
            if conv.bias is not None:
                c.conv.bias.copy_(conv.bias)
        return c


class ConformalReLU(nn.Module):
    """Exact monotone propagation: relu is non-decreasing."""

    def forward(self, z: Interval) -> Interval:
        return monotone_interval(F.relu, z)


class ConformalMaxPool2d(nn.Module):
    """Exact monotone propagation: max-pool is non-decreasing in each input."""

    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, z: Interval) -> Interval:
        return Interval(self.pool(z.lo), self.pool(z.hi))


class ConformalFlatten(nn.Module):
    def forward(self, z: Interval) -> Interval:
        return Interval(z.lo.flatten(1), z.hi.flatten(1))


class ConformalBatchNorm2d(nn.Module):
    """Inference-mode BN is affine: exact interval map, no slack."""

    def __init__(self, num_features: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, z: Interval) -> Interval:
        self.bn.eval()
        w = self.bn.weight / torch.sqrt(self.bn.running_var + self.bn.eps)
        b = self.bn.bias - self.bn.weight * self.bn.running_mean / torch.sqrt(self.bn.running_var + self.bn.eps)
        wv = w.view(1, -1, 1, 1)
        bv = b.view(1, -1, 1, 1)
        Wpos = torch.clamp(wv, min=0.0)
        Wneg = torch.clamp(wv, max=0.0)
        lo = z.lo * Wpos + z.hi * Wneg + bv
        hi = z.hi * Wpos + z.lo * Wneg + bv
        return Interval(lo, hi)

    @classmethod
    def from_bn(cls, bn: nn.BatchNorm2d) -> "ConformalBatchNorm2d":
        c = cls(bn.num_features)
        c.bn.load_state_dict(bn.state_dict())
        return c


class ConformalInput(nn.Module):
    """Input layer: point image -> interval with learned per-channel slack.

    The single statistical step of the whole network lives here: slack is
    trained so that propagated sets cover, and split-conformal calibration
    on the margin score gives the finite-sample guarantee.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.log_slack = nn.Parameter(torch.full((channels,), -3.0))

    def forward(self, x: torch.Tensor) -> Interval:
        s = F.softplus(self.log_slack)
        shape = [1, -1] + [1] * (x.dim() - 2)
        s = s.view(*shape)
        return Interval(x - s, x + s)
