#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

TS_INDEX = PANG / "subcortex" / "brainstem_roi_timeseries" / "brainstem_roi_timeseries_index.csv"

OUT = PANG / "subcortex" / "vta_multivariate_glm"
OUT.mkdir(parents=True, exist_ok=True)

REGRESSORS = {
    "sentence_boundary": PANG / "regressors" / "sentence_boundary_hrf_per_subject",
    "sentence_shift": PANG / "regressors" / "sentence_shift_hrf_per_subject",
    "token_shift": PANG / "regressors" / "shift_hrf_per_subject",
    "pred_error_ar": PANG / "regressors" / "pred_error_ar_hrf_per_subject",
    "pred_error_subspace": PANG / "regressors" / "pred_error_subspace_hrf_per_subject",
    "curvature": PANG / "regressors" / "curvature_hrf_per_subject",
}


def find_regressor(pred, subject, run):
    d = REGRESSORS[pred]
    r = int(str(run).replace("run-", ""))

    candidates = [
        d / f"{subject}_run-{r:02d}.npy",
        d / f"{subject}_run-{r}.npy",
    ]

    for f in candidates:
        if f.exists():
            return f

    hits = list(d.glob(f"*{subject}*run-{r:02d}*.npy"))
    hits += list(d.glob(f"*{subject}*run-{r}*.npy"))

    if hits:
        return sorted(hits)[0]

    return None


def zscore(x):
    x = np.asarray(x, float)
    good = np.isfinite(x)

    y = np.full_like(x, np.nan, dtype=float)

    if good.sum() < 5:
        return y

    sd = np.nanstd(x[good])
    if not np.isfinite(sd) or sd == 0:
        return y

    y[good] = (x[good] - np.nanmean(x[good])) / sd
    return y


def fit_ols(y, X):
    good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y = y[good]
    X = X[good]

    if len(y) < X.shape[1] + 5:
        return None

    X_design = np.column_stack([np.ones(len(y)), X])

    try:
        beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
        resid = y - X_design @ beta
        dof = len(y) - X_design.shape[1]
        sigma2 = np.sum(resid ** 2) / dof
        cov = sigma2 * np.linalg.inv(X_design.T @ X_design)
        se = np.sqrt(np.diag(cov))
        tvals = beta / se
        pvals = 2 * stats.t.sf(np.abs(tvals), dof)
    except Exception:
        return None

    return beta, se, tvals, pvals, dof, len(y)


def run():
    idx = pd.read_csv(TS_INDEX)

    # Primary VTA only. LC is too small and should remain exploratory.
    idx = idx[idx["roi"].isin(["VTA_L", "VTA_R"])].copy()

    rows = []

    predictors = list(REGRESSORS.keys())

    for _, row in idx.iterrows():
        subject = row["subject"]
        run_id = f"{int(row['run']):02d}"
        roi = row["roi"]

        y = np.load(row["timeseries_file"]).squeeze().astype(float)

        reg_arrays = []
        reg_files = {}

        missing = False

        for pred in predictors:
            f = find_regressor(pred, subject, run_id)

            if f is None:
                print(f"[skip] missing {pred}: {subject} run-{run_id}")
                missing = True
                break

            x = np.load(f).squeeze().astype(float)
            reg_arrays.append(x)
            reg_files[pred] = str(f)

        if missing:
            continue

        n = min([len(y)] + [len(x) for x in reg_arrays])

        y_use = zscore(y[:n])
        X_use = np.column_stack([zscore(x[:n]) for x in reg_arrays])

        result = fit_ols(y_use, X_use)

        if result is None:
            print(f"[skip] model failed: {subject} run-{run_id} {roi}")
            continue

        beta, se, tvals, pvals, dof, n_used = result

        # beta[0] is intercept; predictors start at index 1
        for j, pred in enumerate(predictors):
            rows.append({
                "subject": subject,
                "run": run_id,
                "roi": roi,
                "predictor": pred,
                "beta": beta[j + 1],
                "se": se[j + 1],
                "t": tvals[j + 1],
                "p": pvals[j + 1],
                "dof": dof,
                "n_used": n_used,
                "n_predictors": len(predictors),
                "timeseries_file": row["timeseries_file"],
                "regressor_file": reg_files[pred],
            })

        print(f"✅ {subject} run-{run_id} {roi}")

    out = pd.DataFrame(rows)

    out_csv = OUT / "vta_multivariate_glm_by_run.csv"
    out.to_csv(out_csv, index=False)

    print(f"\nWrote {out_csv}")
    print(out.head())


if __name__ == "__main__":
    run()