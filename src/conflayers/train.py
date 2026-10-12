"""Training the confidence compiler.

Objective: minimize expected (surrogate) prediction-set size subject to
empirical coverage >= 1 - alpha on the training batches, via Lagrangian:

    L = E[set_size_surrogate] + lambda * relu((1 - alpha + margin) - coverage)

Only the slack parameters are trained by default (weights frozen from the
compiled point net): the network learns WHERE it is uncertain, layer by
layer, instead of how to classify. Joint weight+slack training is
supported via train_weights=True.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .sets import set_size_surrogate, label_set_from_intervals


class SetSizeTrainer:
    def __init__(self, model, alpha=0.1, lagrangian=10.0, coverage_margin=0.0,
                 lr=1e-2, temperature=1.0, train_weights=False, device="cpu"):
        self.model = model.to(device)
        self.alpha = alpha
        self.lagrangian = lagrangian
        self.coverage_margin = coverage_margin
        self.temperature = temperature
        self.device = device
        if train_weights:
            params = list(model.parameters())
        else:
            params = model.slack_parameters()
        self.opt = torch.optim.Adam(params, lr=lr)

    def step(self, x, y):
        self.model.train()
        x, y = x.to(self.device), y.to(self.device)
        iv = self.model(x)
        size = set_size_surrogate(iv, self.temperature).mean()
        with torch.no_grad():
            mask = label_set_from_intervals(iv, threshold=0.0)
            cov = mask[torch.arange(len(y), device=self.device), y].float().mean()
        target = (1 - self.alpha) + self.coverage_margin
        viol = F.relu(target - cov)
        loss = size + self.lagrangian * viol
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return {"loss": loss.item(), "set_size": size.item(),
                "coverage": cov.item(), "violation": viol.item()}

    def fit(self, loader, epochs, log_every=10):
        hist = []
        for ep in range(epochs):
            agg = {"loss": 0.0, "set_size": 0.0, "coverage": 0.0, "violation": 0.0}
            nb = 0
            for x, y in loader:
                s = self.step(x, y)
                for k in agg:
                    agg[k] += s[k]
                nb += 1
            for k in agg:
                agg[k] /= max(nb, 1)
            agg["epoch"] = ep
            hist.append(agg)
            if ep % log_every == 0:
                print(f"ep {ep}: " + " ".join(f"{k}={v:.4f}" for k, v in agg.items() if k != "epoch"),
                      flush=True)
        return hist
