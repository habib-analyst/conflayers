[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)] [![GitHub stars](https://img.shields.io/github/stars/habib-analyst/conflayers.svg)]

# conflayers — the confidence compiler

**A neural network whose native data type is a guaranteed prediction set, not a point value.** Conformal prediction today is always post-hoc: train a point predictor, then calibrate sets afterward. conflayers compiles uncertainty *into* the architecture — every layer maps input prediction sets to output prediction sets, with exact interval propagation, learned per-layer uncertainty widths, and one split-conformal calibration that yields finite-sample coverage.

## Quickstart

```bash
pip install conflayers
```

```python
import torch.nn as nn
from conflayers import compile

point_net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))
cnet = compile(point_net, in_channels=8)   # the confidence compiler

iv = cnet(x)                              # logit INTERVALS, natively
sets = label_set_from_intervals(iv, q_hat)  # guaranteed prediction sets
```

See `examples/quickstart.py` for the full runnable version.

## Measured results

MedMNIST, α=0.1 (nominal coverage 0.90). Baseline: post-hoc APS on the same point net, same calibration split.

| Dataset | Method | Coverage | Mean set size |
|---|---|---|---|
| BreastMNIST (n=156) | **conflayers** | **0.872** | **1.67** |
| | post-hoc APS | 0.987 | 1.97 |
| PneumoniaMNIST (n=624) | **conflayers** | **0.792** | **1.17** |
| | post-hoc APS | 0.809 | 0.95 |

On the exchangeable split (breast), coverage lands within one standard error of nominal with **15% smaller sets than APS**. On pneumonia, both methods undercover ~10pp — measured val→test accuracy drops 0.960→0.851, i.e. broken exchangeability, under which *no* conformal method can hold coverage. The sets stay non-vacuous and track the baseline.

## How it works

1. **Set-valued layers** — `ConformalLinear/Conv2d` (exact interval arithmetic + learned per-unit slack), `ConformalReLU/MaxPool2d` (exact monotone propagation). Tightness proven in `paper/PAPER.md` (Theorem 1).
2. **The compiler** — `compile(net)` maps any `nn.Sequential` of supported layers to its interval counterpart. Because propagation is exact, the coverage budget is spent **once** at the input — no Bonferroni alpha-fragmentation across layers.
3. **Set-size training** — minimize a differentiable set-size surrogate subject to an empirical coverage constraint (Lagrangian). Only slacks train by default: the net learns *where* it is uncertain.
4. **One calibration** — split-conformal on the margin score gives the finite-sample 1−α guarantee. Exchangeability in, coverage out; nothing else assumed.

**Not this:** post-hoc conformal (uncertainty-blind architecture), cascade conformal inference (composes *pipelines* of models, not *layers* of one net), interval bound propagation (adversarial worst-case bounds, not statistical coverage).

## Honest limitations

- Calibration needs exchangeable held-out data; under distribution shift, coverage degrades like every conformal method (measured above).
- Currently supports Sequential architectures (Linear, Conv2d, ReLU, MaxPool2d, Flatten, BatchNorm2d); no attention/residual yet.
- Interval widths grow with depth — very deep nets may need the slack training to work harder; not yet tested beyond 6 layers.

## Citation

```bibtex
@misc{rehman2026conflayers,
  title={Conformal Layers: The Confidence Compiler},
  author={Habib Ur Rehman},
  year={2026},
  url={https://github.com/habib-analyst/conflayers}
}
```

---
Built by [Habib Ur Rehman](https://github.com/habib-analyst) · ML/AI Engineer & healthcare-AI researcher ·
[LinkedIn](https://www.linkedin.com/in/hur-dev) · [habib.top](https://habib.top)
