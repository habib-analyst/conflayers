"""Prediction sets from logit intervals.

Given logit intervals [l_y, u_y] per class, the prediction set is the
could-be-argmax set: y is included iff u_y >= max_{y'} l_{y'}.
This is the tightest label set consistent with the intervals — any label
outside cannot be the argmax under any realization inside the intervals.

The nonconformity (margin) score for calibration:
    s(x, y) = max_{y'} l_{y'} - u_y
i.e. how far y's logit ceiling falls short of the best competitor floor.
s <= 0 means y could be the argmax. Split-conformal on s gives the
finite-sample 1-alpha guarantee (exchangeability only).
"""
from __future__ import annotations

import torch

from .intervals import Interval


def label_set_from_intervals(iv: Interval, threshold: float = 0.0) -> torch.Tensor:
    """Boolean mask (batch, classes): could-be-argmax labels.

    With threshold q from calibration: include y iff margin_y <= q,
    where margin_y = max_{y'!=y} u_{y'} - l_y. threshold=0 gives the
    raw could-be-argmax set.
    """
    margins = margin_scores(iv)
    return margins <= threshold


def margin_scores(iv: Interval) -> torch.Tensor:
    """Per-class margin scores (batch, classes). Lower = more plausible.

    margin_y = max_{y'} l_{y'} - u_y: how far y's ceiling falls short of
    the best competitor floor. margin_y <= 0 iff y could be the argmax.
    """
    lo, hi = iv.lo, iv.hi
    best_floor = lo.max(dim=1, keepdim=True).values
    return best_floor - hi


def set_size_surrogate(iv: Interval, temperature: float = 1.0) -> torch.Tensor:
    """Differentiable surrogate of expected prediction-set size.

    sum_y sigmoid(margin_y / temperature): soft count of included labels.
    Minimizing this shrinks intervals where the point map is reliable.
    """
    return torch.sigmoid(margin_scores(iv) / temperature).sum(dim=1)


def empirical_coverage(iv: Interval, y: torch.Tensor, threshold: float = 0.0) -> float:
    """Fraction of batch whose true label is in the (thresholded) set."""
    mask = label_set_from_intervals(iv, threshold)
    return mask[torch.arange(len(y)), y].float().mean().item()
