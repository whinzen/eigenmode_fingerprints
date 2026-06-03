#!/usr/bin/env python

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

ENERGY_INDEX = PANG / "subcortex" / "hippocampus_energy" / "hippocampus_energy_index.csv"
OUT_BASE = PANG / "subcortex" / "hippocampus_glm"

REGRESSOR_DIRS = {
    "token_shift": PANG / "regressors" / "shift_hrf_per_subject",
    "pred_error_ar": PANG / "regressors" / "pred_error_ar_hrf_per_subject",
    "pred_error_subspace": PANG / "regressors" / "pred_error_subspace_hrf_per_subject",
    "curvature": PANG / "regressors" / "curvature_hrf_per_subject",
}


def find_regressor(metric, sub, run):
    d = REGRESSOR_DIRS[metric]
    run_num = run.replace("run-", "").lstrip("0")
    run_num_2 = f"{int(run_num):02d}"

    candidates = [
        d / f"{sub}_{run}.npy",
        d / f"{sub}_run-{run_num}.npy",
        d / f"{sub}_run-{run_num_2}.npy",
        d / f"{sub}_task-lppEN_run-{run_num}.npy",
        d / f"{sub}_task-lppEN_run-{run_num_2}.npy",
    ]

    for f in candidates:
        if f.exists():
            return f

    hits = list(d.glob(f"**/*{sub}*run-{run_num}*.npy"))
    hits += list(d.glob(f"**/*{sub}*run-{run_num_2}*.npy"))

    if hits:
        return sorted(hits)[0]

    raise FileNotFoundError(f"No regressor found for {metric}: {sub} {run}")


def zscore(x):
    x = np.asarray(x, float)
    good = np.isfinite(x)
    y = np.zeros_like(x, dtype=float)

    if good.sum() < 3:
        return y

    mu = np.nanmean(x[good])
    sd = np.nanstd(x[good])

    if not np.isfinite(sd) or sd == 0:
        return y

    y[good] = (x[good] - mu) / sd
    return y


def ols_beta(y, x):
    good = np.isfinite(y) & np.isfinite(x)
    y = y[good]
    x = x[good]

    if len(y) < 5 or np.nanstd(x) == 0:
        return np.nan, np.nan, np.nan

    X = np.column_stack([np.ones(len(x)), x])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    resid = y - X @ coef
    dof = len(y) - X.shape[1]

    if dof <= 0:
        return coef[1], np.nan, np.nan

    s2 = np.sum(resid ** 2) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(cov[1, 1])

    if se == 0 or not np.isfinite(se):
        return coef[1], np.nan, np.nan

    tval = coef[1] / se
    pval = 2 * stats.t.sf(np.abs(tval), dof)

    return coef[1], tval, pval


def run_metric(metric):
    out_dir = OUT_BASE / metric
    out_dir.mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(ENERGY_INDEX)
    rows = []

    for _, r in index.iterrows():
        sub = r["sub"]
        run = r["run"]
        hemi = r["hemi"]

        E = np.load(r["energy_file"])  # K x T
        K, T = E.shape

        try:
            reg_file = find_regressor(metric, sub, run)
        except FileNotFoundError as e:
            print(f"[skip] {e}")
            continue

        x = np.load(reg_file).squeeze().astype(float)

        n = min(T, len(x))
        E_use = E[:, :n]
        x_use = zscore(x[:n])

        out_rows = []

        for k in range(K):
            y = zscore(E_use[k])
            beta, tval, pval = ols_beta(y, x_use)

            out_rows.append({
                "sub": sub,
                "run": run,
                "hemi": hemi,
                "mode_k": k,
                "beta": beta,
                "t": tval,
                "p": pval,
                "n_trs": n,
                "regressor_file": str(reg_file),
                "energy_file": r["energy_file"],
            })

        out_df = pd.DataFrame(out_rows)
        out_csv = out_dir / f"{sub}_{run}_hemi-{hemi}_{metric}_hipp_glm.csv"
        out_df.to_csv(out_csv, index=False)

        rows.extend(out_rows)
        print(f"✅ {metric}: {sub} {run} hemi-{hemi}")

    all_df = pd.DataFrame(rows)
    all_csv = out_dir / f"all_{metric}_hipp_glm_rows.csv"
    all_df.to_csv(all_csv, index=False)

    print(f"\n✅ wrote {all_csv}")
    print(all_df.head())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metric",
        choices=["token_shift", "pred_error_ar", "pred_error_subspace", "curvature", "all"],
        default="all",
    )
    args = ap.parse_args()

    metrics = list(REGRESSOR_DIRS.keys()) if args.metric == "all" else [args.metric]

    for metric in metrics:
        run_metric(metric)


if __name__ == "__main__":
    main()