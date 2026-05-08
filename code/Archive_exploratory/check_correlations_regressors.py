import numpy as np
from pathlib import Path

BASE = Path.home() / "eigenmode_fingerprints" / "pang_out" / "regressors"
subj = "sub-EN057"
run = 15

metrics = ["shift", "pred_error_ar", "pred_error_subspace", "curvature"]
xs = []
for m in metrics:
    x = np.load(BASE / f"{m}_hrf_per_subject" / f"{subj}_run-{run}.npy")
    xs.append(x)

X = np.column_stack(xs)
C = np.corrcoef(X, rowvar=False)
print(metrics)
print(np.round(C, 3))