"""Split-conformal calibration on the margin score.

Standard split conformal prediction (Angelopoulos & Bates): given
exchangeable calibration scores s_1..s_n, the threshold
    q_hat = quantile(s_i, ceil((n+1)(1-alpha))/n)
gives P(s_{n+1} <= q_hat) >= 1 - alpha, i.e. the true label's margin
falls below threshold with probability >= 1-alpha — exactly the
statement "true label in the prediction set".

This is the single statistical step of the whole architecture. The
layers contribute structure (exact propagation); calibration contributes
the finite-sample guarantee. No distributional assumptions anywhere.
"""
from __future__ import annotations

import math
import torch

from .intervals import Interval
from .sets import margin_scores, label_set_from_intervals


def calibrate_threshold(scores: torch.Tensor, alpha: float) -> float:
    """Split-conformal quantile threshold for miscoverage level alpha."""
    n = len(scores)
    level = math.ceil((n + 1) * (1 - alpha)) / n
    return torch.quantile(scores, min(level, 1.0)).item()


@torch.no_grad()
def calibrate(model, loader, alpha: float, device="cpu") -> float:
    """Calibrate threshold on margin scores of TRUE labels.

    Returns q_hat. Prediction sets {y: margin_y <= q_hat} then satisfy
    empirical coverage >= 1 - alpha on exchangeable test data.
    """
    model.eval()
    all_s = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        iv = model(x)
        m = margin_scores(iv)
        all_s.append(m[torch.arange(len(y)), y])
    scores = torch.cat(all_s)
    return calibrate_threshold(scores, alpha)


@torch.no_grad()
def conformal_sets(model, loader, q_hat: float, device="cpu"):
    """Evaluate: coverage and mean set size at calibrated threshold."""
    model.eval()
    cov_n, cov_d, size_sum, n = 0, 0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        iv = model(x)
        mask = label_set_from_intervals(iv, q_hat)
        cov_n += mask[torch.arange(len(y)), y].sum().item()
        cov_d += len(y)
        size_sum += mask.sum().item()
        n += len(y)
    return {"coverage": cov_n / cov_d, "mean_set_size": size_sum / n, "n": n}
