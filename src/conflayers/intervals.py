"""Interval arithmetic primitives.

An Interval is a pair (lo, hi) of tensors with lo <= hi elementwise,
representing the set of all tensors between them. All layer maps are
*exact*: the output interval is the tightest interval containing the
image of the input interval under the point map. Exactness is what makes
the composition theorem hold (see paper, Theorem 1).
"""
from __future__ import annotations

import torch


class Interval:
    """Closed interval [lo, hi] over tensors, lo <= hi elementwise."""

    def __init__(self, lo: torch.Tensor, hi: torch.Tensor):
        if torch.any(lo > hi):
            raise ValueError("lo > hi somewhere")
        self.lo = lo
        self.hi = hi

    @classmethod
    def point(cls, x: torch.Tensor) -> "Interval":
        """Degenerate interval: a single point."""
        return cls(x, x)

    @property
    def shape(self):
        return self.lo.shape

    @property
    def width(self) -> torch.Tensor:
        return self.hi - self.lo

    @property
    def mid(self) -> torch.Tensor:
        return 0.5 * (self.lo + self.hi)

    def __repr__(self):
        return f"Interval(shape={tuple(self.shape)}, mean_width={self.width.mean().item():.4f})"


def affine_interval(W: torch.Tensor, b: torch.Tensor | None, z: Interval) -> Interval:
    """Exact interval image of z -> W z + b.

    For each output j: lo_j = sum_{W>0} W_jk lo_k + sum_{W<0} W_jk hi_k + b_j
    (and symmetrically for hi). Tight because the map is affine in each input.
    """
    Wpos = torch.clamp(W, min=0.0)
    Wneg = torch.clamp(W, max=0.0)
    lo = z.lo @ Wpos.T + z.hi @ Wneg.T
    hi = z.hi @ Wpos.T + z.lo @ Wneg.T
    if b is not None:
        lo = lo + b
        hi = hi + b
    return Interval(lo, hi)


def monotone_interval(f, z: Interval) -> Interval:
    """Exact interval image of a non-decreasing elementwise map."""
    return Interval(f(z.lo), f(z.hi))
