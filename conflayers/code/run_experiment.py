"""Main experiment: does the compiled net achieve nominal coverage end-to-end,
with mean set size beating post-hoc APS? Resumable per stage."""
import json
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from conflayers import compile, ConformalNet
from conflayers.data import load_medmnist
from conflayers.sets import label_set_from_intervals
from conflayers.calibrate import calibrate, conformal_sets
from conflayers.train import SetSizeTrainer

DEVICE = "cpu"
ALPHA = 0.1
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)


def point_cnn(n_classes):
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(),
        nn.Linear(64, n_classes),
    )


def train_point(net, loader, epochs=12):
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for ep in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            loss = F.cross_entropy(net(x), y)
            loss.backward()
            opt.step()


@torch.no_grad()
def accuracy(net, loader):
    net.eval()
    c, n = 0, 0
    for x, y in loader:
        c += (net(x).argmax(1) == y).sum().item()
        n += len(y)
    return c / n


@torch.no_grad()
def aps_sets(net, loader, q_hat):
    """Post-hoc APS baseline: softmax scores, cumulative-probability sets."""
    net.eval()
    cov_n, cov_d, size = 0, 0, 0.0
    for x, y in loader:
        p = F.softmax(net(x), dim=1).sort(descending=True)
        probs, order = p.values, p.indices
        cumsum = probs.cumsum(1)
        # APS score of true label = cumsum at its rank
        rank = (order == y.unsqueeze(1)).float().argmax(1)
        scores = cumsum[torch.arange(len(y)), rank]
        # set = labels with cumsum <= q
        mask = cumsum <= q_hat
        # map back to label space
        lab_mask = torch.zeros_like(mask, dtype=torch.bool)
        lab_mask.scatter_(1, order, mask)
        cov_n += lab_mask[torch.arange(len(y)), y].sum().item()
        cov_d += len(y)
        size += lab_mask.sum().item()
    return {"coverage": cov_n / cov_d, "mean_set_size": size / cov_d}


@torch.no_grad()
def aps_calibrate(net, loader, alpha):
    import math
    net.eval()
    ss = []
    for x, y in loader:
        p = F.softmax(net(x), dim=1).sort(descending=True)
        probs, order = p.values, p.indices
        cumsum = probs.cumsum(1)
        rank = (order == y.unsqueeze(1)).float().argmax(1)
        ss.append(cumsum[torch.arange(len(y)), rank])
    s = torch.cat(ss)
    n = len(s)
    return torch.quantile(s, min(math.ceil((n + 1) * (1 - alpha)) / n, 1.0)).item()


def run_dataset(name, force=False):
    done = os.path.join(RES, f"{name}.done")
    if os.path.exists(done) and not force:
        return json.load(open(os.path.join(RES, f"{name}.json")))
    print(f"=== {name} ===", flush=True)
    data = load_medmnist(name)
    nc = data["n_classes"]

    # Stage 1: point net
    wpath = os.path.join(RES, f"{name}.point.pt")
    net = point_cnn(nc)
    if os.path.exists(wpath) and not force:
        net.load_state_dict(torch.load(wpath, map_location="cpu"))
    else:
        train_point(net, data["train"], epochs=12)
        torch.save(net.state_dict(), wpath)
    acc = accuracy(net, data["test"])
    print(f"point test acc: {acc:.4f}", flush=True)

    # Stage 2: compile + train slacks
    spath = os.path.join(RES, f"{name}.slack.pt")
    cnet = compile(net, in_channels=1)
    if os.path.exists(spath) and not force:
        cnet.load_state_dict(torch.load(spath, map_location="cpu"))
    else:
        trainer = SetSizeTrainer(cnet, alpha=ALPHA, lagrangian=20.0, lr=5e-3)
        trainer.fit(data["train"], epochs=25, log_every=25)
        torch.save(cnet.state_dict(), spath)

    # Stage 3: calibrate + evaluate compiled net
    q = calibrate(cnet, data["val"], ALPHA)
    ev = conformal_sets(cnet, data["test"], q)

    # Stage 4: post-hoc APS baseline on the same point net
    q_aps = aps_calibrate(net, data["val"], ALPHA)
    ev_aps = aps_sets(net, data["test"], q_aps)

    out = {
        "dataset": name, "alpha": ALPHA, "point_test_acc": acc,
        "compiled": {"q_hat": q, **ev},
        "aps_baseline": {"q_hat": q_aps, **ev_aps},
    }
    json.dump(out, open(os.path.join(RES, f"{name}.json"), "w"), indent=1)
    open(done, "w").write("done")
    print(json.dumps(out, indent=1), flush=True)
    return out


if __name__ == "__main__":
    force = "--force" in sys.argv
    results = {}
    for name in ["breastmnist", "pneumoniamnist"]:
        results[name] = run_dataset(name, force=force)
    json.dump(results, open(os.path.join(RES, "summary.json"), "w"), indent=1)
    open(os.path.join(RES, "PIPELINE_DONE"), "w").write("done")
    print("PIPELINE_DONE", flush=True)
