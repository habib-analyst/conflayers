"""The confidence compiler: nn.Module -> ConformalNet.

Maps a standard point-valued network (nn.Sequential of supported layers)
to its set-valued counterpart. Because every layer map is an EXACT
interval propagation (no statistical approximation per layer), the
coverage budget is spent ONCE — at the input layer — instead of
Bonferroni-splitting alpha across layers. That is the compiler's core
advantage over naive per-layer conformalization: no alpha fragmentation,
no vacuous union bounds.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .intervals import Interval
from .layers import (
    ConformalLinear,
    ConformalConv2d,
    ConformalReLU,
    ConformalMaxPool2d,
    ConformalFlatten,
    ConformalBatchNorm2d,
    ConformalInput,
)

_MAPPING = {
    nn.Linear: ConformalLinear.from_linear,
    nn.Conv2d: ConformalConv2d.from_conv,
    nn.ReLU: lambda m: ConformalReLU(),
    nn.MaxPool2d: lambda m: ConformalMaxPool2d(m.kernel_size, stride=m.stride, padding=m.padding),
    nn.Flatten: lambda m: ConformalFlatten(),
    nn.BatchNorm2d: ConformalBatchNorm2d.from_bn,
}

SUPPORTED = tuple(_MAPPING.keys())


class ConformalNet(nn.Module):
    """A compiled network: forward maps Tensor -> Interval over logits."""

    def __init__(self, layers: nn.ModuleList, in_channels: int):
        super().__init__()
        self.input_layer = ConformalInput(in_channels)
        self.layers = layers

    def forward(self, x: torch.Tensor) -> Interval:
        z = self.input_layer(x)
        for layer in self.layers:
            z = layer(z)
        return z

    def point_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Midpoint of the output interval (the compiled point prediction)."""
        return self.forward(x).mid

    def slack_parameters(self):
        return [p for n, p in self.named_parameters() if "log_slack" in n]


def compile(model: nn.Module, in_channels: int) -> ConformalNet:  # noqa: A001
    """Compile a point network into its set-valued counterpart.

    Accepts nn.Sequential (or a ModuleList) of supported layers.
    Weights are copied; slacks start near-tight and are learned.
    """
    seq = model if isinstance(model, (nn.Sequential, nn.ModuleList)) else None
    if seq is None:
        mods = list(model.children())
        if not mods:
            raise ValueError("compile needs a Sequential/ModuleList or a module with children")
    else:
        mods = list(seq)

    clayers = []
    for m in mods:
        mapped = None
        for cls, fn in _MAPPING.items():
            if isinstance(m, cls):
                mapped = fn(m)
                break
        if mapped is None:
            raise TypeError(f"conflayers.compile: unsupported layer {type(m).__name__}; "
                            f"supported: {[c.__name__ for c in SUPPORTED]}")
        clayers.append(mapped)
    return ConformalNet(nn.ModuleList(clayers), in_channels)
