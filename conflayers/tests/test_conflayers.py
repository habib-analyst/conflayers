"""Tests for conflayers. The exactness tests are the mathematical core:
interval propagation must TIGHTLY contain the point map's image."""
import torch
import torch.nn as nn

from conflayers.intervals import Interval, affine_interval, monotone_interval
from conflayers.layers import ConformalLinear, ConformalConv2d, ConformalReLU, ConformalMaxPool2d
from conflayers.compiler import compile, ConformalNet
from conflayers.sets import label_set_from_intervals, margin_scores, empirical_coverage
from conflayers.calibrate import calibrate_threshold

torch.manual_seed(0)


def test_affine_interval_exact():
    """Interval affine map tightly contains brute-force sampled image."""
    W = torch.randn(4, 6)
    b = torch.randn(4)
    lo = torch.randn(32, 6) - 1.0
    hi = lo + torch.rand(32, 6) * 2.0
    z = Interval(lo, hi)
    out = affine_interval(W, b, z)
    # brute force: sample corners (exact for affine: extrema at corners)
    import itertools
    for i in range(32):
        corners = torch.stack([
            torch.where(torch.tensor(c, dtype=torch.bool), hi[i], lo[i])
            for c in itertools.product([0, 1], repeat=6)
        ])
        vals = corners @ W.T + b
        assert torch.all(vals.min(0).values >= out.lo[i] - 1e-5)
        assert torch.all(vals.max(0).values <= out.hi[i] + 1e-5)
        # tightness: bounds attained
        assert torch.allclose(vals.min(0).values, out.lo[i], atol=1e-5)
        assert torch.allclose(vals.max(0).values, out.hi[i], atol=1e-5)


def test_monotone_exact():
    a = torch.randn(16, 8) - 1
    b = torch.randn(16, 8) + 1
    z = Interval(torch.minimum(a, b), torch.maximum(a, b))
    out = monotone_interval(torch.relu, z)
    assert torch.allclose(out.lo, torch.relu(z.lo))
    assert torch.allclose(out.hi, torch.relu(z.hi))


def test_conv_interval_contains_point():
    conv = ConformalConv2d(1, 4, 3, padding=1)
    x = torch.randn(5, 1, 8, 8)
    iv = conv(Interval.point(x))
    with torch.no_grad():
        pt = conv.conv(x)
    assert torch.all(pt >= iv.lo - 1e-5) and torch.all(pt <= iv.hi + 1e-5)


def test_maxpool_exact():
    p = ConformalMaxPool2d(2, stride=2)
    a = torch.randn(4, 2, 6, 6)
    b = torch.randn(4, 2, 6, 6)
    z = Interval(torch.minimum(a, b), torch.maximum(a, b))
    out = p(z)
    assert torch.allclose(out.lo, p.pool(z.lo))
    assert torch.allclose(out.hi, p.pool(z.hi))


def test_compile_maps_weights():
    net = nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(4 * 14 * 14, 2),
    )
    cn = compile(net, in_channels=1)
    assert isinstance(cn, ConformalNet)
    x = torch.randn(3, 1, 28, 28)
    iv = cn(x)
    assert iv.lo.shape == (3, 2) and iv.hi.shape == (3, 2)
    # midpoint ≈ point net output (slack starts near-tight)
    with torch.no_grad():
        pt = net(x)
    assert torch.allclose(iv.mid, pt, atol=0.3)


def test_compile_rejects_unsupported():
    net = nn.Sequential(nn.Linear(4, 4), nn.Sigmoid())
    try:
        compile(net, in_channels=1)
        raise AssertionError("should have raised")
    except TypeError:
        pass


def test_label_set_could_be_argmax():
    # class 0 could beat class 1 iff u0 >= l1
    iv = Interval(torch.tensor([[0.1, 0.0]]), torch.tensor([[0.5, 0.4]]))
    mask = label_set_from_intervals(iv, threshold=0.0)
    assert mask[0, 0] and mask[0, 1]  # u0=0.5>=l1=0.0, u1=0.4>=l0=0.1
    iv2 = Interval(torch.tensor([[0.5, 0.0]]), torch.tensor([[0.6, 0.4]]))
    mask2 = label_set_from_intervals(iv2, threshold=0.0)
    assert mask2[0, 0] and not mask2[0, 1]  # u1=0.4 < l0=0.5


def test_margin_scores_sign():
    iv = Interval(torch.tensor([[0.5, 0.0]]), torch.tensor([[0.6, 0.4]]))
    m = margin_scores(iv)
    assert m[0, 0].item() < 0 < m[0, 1].item()  # class 0 unambiguous


def test_calibration_threshold_valid():
    """Split-conformal threshold gives >= 1-alpha coverage on fresh exchangeable data."""
    torch.manual_seed(1)
    alpha = 0.1
    cal = torch.randn(500).abs()
    q = calibrate_threshold(cal, alpha)
    test = torch.randn(10000).abs()
    assert (test <= q).float().mean().item() >= 1 - alpha - 0.02


def test_slack_starts_tight():
    lin = ConformalLinear(4, 4)
    assert (lin.slack < 0.1).all()


def test_interval_rejects_inverted():
    try:
        Interval(torch.ones(2), torch.zeros(2))
        raise AssertionError("should have raised")
    except ValueError:
        pass
