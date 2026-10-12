"""conflayers: the confidence compiler.

A neural network whose native data type is a GUARANTEED PREDICTION SET,
not a point value. Uncertainty compiled into the architecture instead
of bolted on after training.

Pipeline: compile(point_net) -> ConformalNet with interval-valued forward
pass -> train slacks to minimize set size under coverage -> split-conformal
calibration on the margin score -> prediction sets with finite-sample
coverage guarantees.
"""
from .intervals import Interval
from .layers import (
    ConformalLinear,
    ConformalConv2d,
    ConformalReLU,
    ConformalMaxPool2d,
    ConformalFlatten,
    ConformalBatchNorm2d,
)
from .compiler import compile, ConformalNet, SUPPORTED
from .sets import label_set_from_intervals, margin_scores, set_size_surrogate
from .calibrate import calibrate_threshold, conformal_sets
from .train import SetSizeTrainer

__version__ = "0.1.0"
__all__ = [
    "Interval",
    "ConformalLinear",
    "ConformalConv2d",
    "ConformalReLU",
    "ConformalMaxPool2d",
    "ConformalFlatten",
    "ConformalBatchNorm2d",
    "compile",
    "ConformalNet",
    "SUPPORTED",
    "label_set_from_intervals",
    "margin_scores",
    "set_size_surrogate",
    "calibrate_threshold",
    "conformal_sets",
    "SetSizeTrainer",
]
