"""Quickstart: compile a point net into a set-valued net and get guaranteed sets."""
import torch
import torch.nn as nn
from conflayers import compile
from conflayers.calibrate import calibrate_threshold
from conflayers.sets import label_set_from_intervals, margin_scores

torch.manual_seed(0)

# 1. A regular point network
point_net = nn.Sequential(
    nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3),
)

# 2. Compile it: every layer becomes set-valued (intervals in, intervals out)
cnet = compile(point_net, in_channels=8)

# 3. Forward pass natively returns a logit INTERVAL per class
x = torch.randn(4, 8)
iv = cnet(x)
print("logit intervals:\n lo =", iv.lo.round(decimals=2), "\n hi =", iv.hi.round(decimals=2))

# 4. Calibrate once on held-out margin scores -> finite-sample 1-alpha guarantee
cal_scores = margin_scores(cnet(torch.randn(200, 8)))[:, 0]  # toy scores
q_hat = calibrate_threshold(cal_scores.abs(), alpha=0.1)

# 5. Prediction sets, guaranteed
sets = label_set_from_intervals(iv, threshold=q_hat)
print("prediction sets (one row per input):\n", sets)
print("mean set size:", sets.float().sum(1).mean().item())
