#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path.home() / "eigenmode_fingerprints" / "pang_out"
OUT = BASE / "paper_tables"
OUT.mkdir(parents=True, exist_ok=True)

PAIRS = [
    ("shift", "pred_error_ar", "Prediction error"),
    ("shift", "pred_error_subspace", "Subspace exit"),
    ("shift", "curvature", "Curvature"),
]

def load_pair_curve(pair_name, metric_name):
    left = BASE / f"group_{pair_name}_glm" / f"group_{pair_name}_{metric_name}_hemi-L_by_mode_subject_level.csv"
    right = BASE / f"group_{pair_name}_glm" / f"group_{pair_name}_{metric_name}_hemi-R_by_mode_subject_level.csv"

    dl = pd.read_csv(left)
    dr = pd.read_csv(right)

    merged = dl.merge(dr, on="mode_k", suffixes=("_L", "_R"))
    merged["beta_mean"] = (merged["beta_mean_L"] + merged["beta_mean_R"]) / 2.0
    return merged["beta_mean"].values

def safe_corr(a, b):
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 3:
        return np.nan
    return np.corrcoef(a[good], b[good])[0, 1]

def main():
    rows = []

    for base_metric, extra_metric, label in PAIRS:
        pair_name = f"{base_metric}__plus__{extra_metric}_resid"

        y_base = load_pair_curve(pair_name, base_metric)
        y_extra = load_pair_curve(pair_name, f"{extra_metric}_resid")

        # stats
        r = safe_corr(y_base, y_extra)

        base_mean_abs = np.nanmean(np.abs(y_base))
        extra_mean_abs = np.nanmean(np.abs(y_extra))

        base_max_abs = np.nanmax(np.abs(y_base))
        extra_max_abs = np.nanmax(np.abs(y_extra))

        ratio = extra_max_abs / base_max_abs if base_max_abs > 0 else np.nan

        rows.append({
            "Metric (residualized)": label,
            "r (raw profile)": round(r, 3),
            "Mean |β| (shift)": round(base_mean_abs, 2),
            "Mean |β| (metric)": round(extra_mean_abs, 2),
            "Max |β| ratio": round(ratio, 2),
        })

    df = pd.DataFrame(rows)

    out_csv = OUT / "table_residualized_amplitudes_clean.csv"
    df.to_csv(out_csv, index=False)

    print("\n✅ Clean table:\n")
    print(df.to_string(index=False))
    print(f"\n✅ wrote {out_csv}")

if __name__ == "__main__":
    main()