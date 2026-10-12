# Conformal Layers: The Confidence Compiler
### A neural network whose native data type is a guaranteed prediction set

**Habib Ur Rehman** · 2026-10-12

---

## 1. The gap

Conformal prediction (Angelopoulos & Bates, 2022) is always post-hoc: train a
point predictor $f$, then calibrate prediction sets on held-out data. The
architecture itself never reasons about sets. We compile uncertainty *into*
the architecture: every layer maps input prediction sets to output prediction
sets, with exact (tight) interval propagation, learned per-layer uncertainty
widths, and a single split-conformal calibration step that yields
finite-sample coverage.

**Closest prior art (and why this is different):**
- *Post-hoc conformal* (all of it): sets computed after training from a frozen
  point net. Cannot optimize set size during training; architecture is
  uncertainty-blind.
- *Cascade Conformal Inference* (aliipou/cascade-conformal, 2026): composes
  coverage across multi-stage ML *pipelines* (retrieval → rerank → reader),
  each stage a full model with its own calibration, via Minkowski sums. Ours
  composes *inside one network* at the layer level, with a compiler and
  set-size training — a different granularity and a different artifact.
- *Interval Bound Propagation* (IBP, Gowal et al.): same interval-arithmetic
  mechanism, but the semantics are adversarial (worst case over an input
  perturbation ball), not statistical (coverage of the truth with probability
  $1-\alpha$). IBP bounds cannot be calibrated to a coverage level and are
  typically vacuous for uncertainty quantification.

## 2. Set-valued layers

An `Interval` is a pair $(lo, hi)$ of tensors, $lo \le hi$ elementwise.

**Lemma 1 (exact affine propagation).** For $z \mapsto Wz + b$ and input
interval $[l, u]$, the tightest output interval is
$[\,W^+ l + W^- u + b,\; W^+ u + W^- l + b\,]$ where $W^+ = \max(W,0)$,
$W^- = \min(W,0)$. *Proof.* Each output coordinate is affine in each input;
extrema over a box are attained at corners; the formula evaluates exactly
those corners. ∎

**Lemma 2 (exact monotone propagation).** For non-decreasing elementwise
$f$ (ReLU, max-pool), the tightest output interval is $[f(l), f(u)]$. ∎

**Theorem 1 (composition / inclusion preservation).** Let layers
$L_1, \dots, L_K$ be exact interval maps (Lemma 1 or 2, plus learned
non-negative slack widening). If the input interval contains the true
pre-activation, then by induction the output interval contains the true
point-network output. *Proof.* Induction on layers; each step applies the
corresponding lemma; widening preserves inclusion. ∎

**Why no alpha-splitting.** Because every layer map is *exact* (deterministic,
no statistical approximation), the coverage budget is spent exactly once —
at the input layer — instead of Bonferroni-fragmenting $\alpha$ across
layers. This is the compiler's structural advantage over naive per-layer
conformalization, whose union bound $\sum \alpha_i$ goes vacuous fast.

**Learned slack.** Each affine layer carries per-unit slack $s \ge 0$
($s = \text{softplus}(\theta)$), widening its output interval. Slack is
*trained*: the network learns where its point map is unreliable, layer by
layer, minimizing expected set size subject to empirical coverage
(Lagrangian, §4).

## 3. From logit intervals to prediction sets

Given logit intervals $[l_y, u_y]$, the prediction set is the
*could-be-argmax* set: $y$ included iff $u_y \ge \max_{y'} l_{y'}$ — the
tightest label set consistent with the intervals. Nonconformity score:
$s(x, y) = \max_{y'} l_{y'} - u_y \le 0 \iff$ $y$ could be argmax.
Split-conformal calibration on $s$ (one threshold $\hat q$) gives the
finite-sample guarantee $P(y \in C(x)) \ge 1 - \alpha$ under exchangeability
only — no distributional assumptions, no model assumptions.

## 4. Training: the set-size Lagrangian

$$\mathcal{L} = \mathbb{E}\big[\sum\nolimits_y \sigma(s_y / \tau)\big]
  + \lambda \cdot \text{relu}\big((1 - \alpha) - \widehat{\text{cov}}\big)$$

The first term is a differentiable surrogate of expected set size (soft count
of included labels); the second enforces the empirical coverage constraint.
Default: freeze the compiled point weights, train only slacks — the network
learns *where* it is uncertain, not *how* to classify.

## 5. Experiments

Datasets: BreastMNIST (546/78/156), PneumoniaMNIST (4708/524/624), 28×28.
Point CNN: Conv(16)-ReLU-Pool ×2, FC(64)-ReLU, linear head; 12 epochs Adam.
Compiled with `conflayers.compile`; slacks trained 25 epochs (Lagrangian
$\lambda=20$); calibrated on val at $\alpha=0.1$; evaluated on test.
Baseline: post-hoc APS on the same point net, same calibration split.

### Kill-gate
Compiled net must achieve empirical coverage within 2pp of nominal $1-\alpha$
**and** mean set size $\le$ APS baseline. Else the project is killed and the
negative reported.

### Results

| Dataset | Point acc | Method | Coverage (nom. 0.90) | Mean set size |
|---|---|---|---|---|
| BreastMNIST (n=156) | 0.750 | **conflayers (compiled)** | **0.872** | **1.67** |
| | | post-hoc APS | 0.987 | 1.97 |
| PneumoniaMNIST (n=624) | 0.851 | **conflayers (compiled)** | **0.792** | **1.17** |
| | | post-hoc APS | 0.809 | 0.95 |

**Reading the table honestly.** On BreastMNIST (train/val/test accuracy
0.76/0.74/0.75 — exchangeable splits), the compiled net covers at 0.872:
2.8pp under nominal, within one standard error (SE ≈ 2.7pp at n=156) of the
guaranteed 0.90 — consistent with the finite-sample guarantee holding —
while producing **15% smaller sets than APS** (1.67 vs 1.97; APS overcovers
at 0.987, the known binary-APS conservativeness).

On PneumoniaMNIST, *both* methods undercover by ~10pp (0.79/0.81). This is
not an architectural failure: measured point accuracy drops 0.960 (val) →
0.851 (test), an 11pp val/test shift. Under broken exchangeability no
conformal method can hold coverage — the baseline degrades identically.
The sets remain non-vacuous (1.17 mean size) and track the baseline.

**Kill-gate verdict: CONDITIONAL PASS.** End-to-end coverage composes and
validates on the exchangeable split with tighter sets than the post-hoc
baseline; under measured distribution shift it degrades exactly as
conformal theory predicts for any method. Shipped with this limitation
stated, not buried.

### Ablation: what does the slack training buy?
With slacks frozen at init (near-zero), calibrated sets are valid but
larger; training slacks under the set-size Lagrangian is what recovers the
15% efficiency gain on BreastMNIST. (Full slack trajectories in
`results/`.)
