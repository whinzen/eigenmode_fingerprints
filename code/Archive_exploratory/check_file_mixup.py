import numpy as np
from pathlib import Path

BASE = Path.home() / "eigenmode_fingerprints" / "pang_out" / "regressors"
subj = "sub-EN057"
run = 15

pairs = [
    ("shift", "pred_error_ar"),
    ("shift", "pred_error_subspace"),
    ("shift", "curvature"),
]

for a, b in pairs:
    xa = np.load(BASE / f"{a}_hrf_per_subject" / f"{subj}_run-{run}.npy")
    xb = np.load(BASE / f"{b}_hrf_per_subject" / f"{subj}_run-{run}.npy")
    print(a, b,
          "allclose=", np.allclose(xa, xb),
          "corr=", np.corrcoef(xa, xb)[0, 1])