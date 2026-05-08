#!/usr/bin/env python

import pandas as pd
from pathlib import Path

BASE = Path.home() / "eigenmode_fingerprints"
IN_DIR = BASE / "pang_out" / "paper_tables"
OUT = IN_DIR / "table_empirical_lowpass_reconstruction_summary.csv"

LABELS = {
    "sentence_shift": "Sentence shift",
    "boundary": "Sentence boundary",
    "token_shift": "Token shift",
    "pred_error_ar": "Prediction error",
    "pred_error_subspace": "Subspace exit",
    "curvature": "Curvature",
}

ORDER = [
    "sentence_shift",
    "boundary",
    "token_shift",
    "pred_error_ar",
    "pred_error_subspace",
    "curvature",
]

data = {}

for f in sorted(IN_DIR.glob("figure_empirical_full_lowpass_residual_*_K20_stats.csv")):
    df = pd.read_csv(f)
    metric = df["metric"].iloc[0]
    data[metric] = df

rows = []

for metric in ORDER:
    if metric not in data:
        print(f"[warn] missing {metric}")
        continue

    df = data[metric]

    full = df[df["comparison"] == "empirical_vs_full_retained"].iloc[0]
    low = df[df["comparison"] == "empirical_vs_lowpass"].iloc[0]

    rows.append({
        "Regressor": LABELS.get(metric, metric),
        "Full r": round(full["r"], 3),
        "Full R²": round(full["r2"], 3),
        "K=20 r": round(low["r"], 3),
        "K=20 R²": round(low["r2"], 3),
    })

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)

print("✅ wrote", OUT)
print(out.to_string(index=False))