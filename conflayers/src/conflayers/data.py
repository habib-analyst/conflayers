"""MedMNIST .npz loading (28x28)."""
from __future__ import annotations

import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

MEDMNIST_DIR = os.path.expanduser("~/.medmnist")


def load_medmnist(name: str, batch_size=128):
    path = os.path.join(MEDMNIST_DIR, f"{name}.npz")
    d = np.load(path)
    def t(k):
        x = torch.from_numpy(d[k]).float()
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return x.permute(0, 1, 3, 2) if False else x  # keep (N,1,28,28)
    Xtr, ytr = t("train_images") / 255.0, torch.from_numpy(d["train_labels"]).long().squeeze()
    Xv, yv = t("val_images") / 255.0, torch.from_numpy(d["val_labels"]).long().squeeze()
    Xte, yte = t("test_images") / 255.0, torch.from_numpy(d["test_labels"]).long().squeeze()
    mk = lambda X, y, sh: DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=sh)
    return {"train": mk(Xtr, ytr, True), "val": mk(Xv, yv, False), "test": mk(Xte, yte, False),
            "n_classes": int(ytr.max().item()) + 1, "in_channels": 1}
