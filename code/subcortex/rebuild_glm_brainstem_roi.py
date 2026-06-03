#!/usr/bin/env python

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy import stats


BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

TS_INDEX = PANG / "subcortex" / "brainstem_roi_timeseries" / "brainstem_roi_timeseries_index.csv"
OUT = PANG / "subcortex" / "brainstem_roi_glm"
OUT.mkdir(parents=True, exist_ok=True)

REGRESSOR_DIRS = {
    "sentence_boundary": PANG / "regressors" / "sentence_boundary_hrf_per_subject",
    "sentence_shift": PANG / "regressors" / "sentence_shift_hrf_per_subject",
    "token_shift": PANG / "regressors" / "shift_hrf_per_subject",
    "pred_error_ar": PANG / "regressors" / "pred_error_ar_hrf_per_subject",
    "pred_error_subspace": PANG / "regressors" / "pred_error_subspace_hrf_per_subject",
    "curvature": PANG / "regressors" / "curvature_hrf_per_subject",
    # Add later when built:
    # "surprisal": PANG / "regressors" / "surprisal_hrf_per_subject",
    # "entropy": PANG / "regressors" / "entropy_hrf_per_subject",
    # "entropy_reduction": PANG / "regressors" / "entropy_reduction_hrf_per_subject",
}


def find_regressor(metric, sub, run):
    d = REGRESSOR_DIRS[metric]
    r = int(str(run).replace("run-", ""))

    candidates = [
        d / f"{sub}_run-{r:02d}.npy",
        d / f"{sub}_run-{r}.npy",
    ]

    for f in candidates:
        if f.exists():
            return f

    return None


def zscore(x):
    x = np.asarray(x, float)
    y = np.full_like(x, np.nan)
    good = np.isfinite(x)

    if good.sum() < 3:
        return y

    sd = np.nanstd(x[good])
    if sd == 0 or not np.isfinite(sd):
        return y

    y[good] = (x[good] - np.nanmean(x[good])) / sd
    return y


def ols_beta(y, x):
    n = min(len(y), len(x))
    y = zscore(y[:n])
    x = zscore(x[:n])

    good = np.isfinite(y) & np.isfinite(x)
    y = y[good]
    x = x[good]

    if len(y) < 10 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, np.nan, np.nan, len(y)

    X = np.column_stack([np.ones(len(x)), x])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    resid = y - X @ coef
    dof = len(y) - X.shape[1]

    s2 = np.sum(resid ** 2) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(cov[1, 1])

    tval = coef[1] / se
    pval = 2 * stats.t.sf(abs(tval), dof)

    return coef[1], tval, pval, len(y)


def run_metric(metric):
    idx = pd.read_csv(TS_INDEX)
    rows = []

    for _, r in idx.iterrows():
        sub = r["subject"]
        run = r["run"]
        roi = r["roi"]

        reg_file = find_regressor(metric, sub, run)

        if reg_file is None:
            print(f"[skip] missing regressor: {metric} {sub} run-{run}")
            continue

        y = np.load(r["timeseries_file"]).squeeze().astype(float)
        x = np.load(reg_file).squeeze().astype(float)

        beta, tval, pval, n_used = ols_beta(y, x)

        rows.append({
            "subject": sub,
            "run": f"{int(run):02d}",
            "roi": roi,
            "predictor": metric,
            "beta": beta,
            "t": tval,
            "p": pval,
            "n_used": n_used,
            "n_voxels": r["n_voxels"],
            "timeseries_file": r["timeseries_file"],
            "regressor_file": str(reg_file),
        })

        print(f"✅ {metric}: {sub} run-{int(run):02d} {roi}")

    df = pd.DataFrame(rows)
    out_csv = OUT / f"brainstem_roi_glm_{metric}.csv"
    df.to_csv(out_csv, index=False)

    print(f"\nWrote {out_csv}")
    print(df.head())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=list(REGRESSOR_DIRS.keys()) + ["all"], default="all")
    args = ap.parse_args()

    metrics = list(REGRESSOR_DIRS.keys()) if args.metric == "all" else [args.metric]

    for metric in metrics:
        run_metric(metric)


if __name__ == "__main__":
    main()