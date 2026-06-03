#!/usr/bin/env python

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"
DATA_ROOT = BASE / "data" / "empirical"

TRAJ_DIR = PANG / "subcortex" / "hippocampus_trajectory"
OUT = PANG / "subcortex" / "hipp_trajectory_mean_covariate"
OUT.mkdir(parents=True, exist_ok=True)

HIP_LABELS = {"L": 17, "R": 53}

REGRESSOR_DIRS = {
    "token_shift": PANG / "regressors" / "shift_hrf_per_subject",
    "pred_error_ar": PANG / "regressors" / "pred_error_ar_hrf_per_subject",
    "pred_error_subspace": PANG / "regressors" / "pred_error_subspace_hrf_per_subject",
    "curvature": PANG / "regressors" / "curvature_hrf_per_subject",
}


def zscore(x):
    x = np.asarray(x, float)
    y = np.full_like(x, np.nan, dtype=float)
    good = np.isfinite(x)

    if good.sum() < 3:
        return y

    sd = np.nanstd(x[good])
    if sd == 0 or not np.isfinite(sd):
        return y

    y[good] = (x[good] - np.nanmean(x[good])) / sd
    return y


def ols_two_predictors(y, x1, x2):
    n = min(len(y), len(x1), len(x2))

    y = zscore(y[:n])
    x1 = zscore(x1[:n])
    x2 = zscore(x2[:n])

    good = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2)

    y = y[good]
    x1 = x1[good]
    x2 = x2[good]

    if len(y) < 10:
        return [np.nan] * 7

    X = np.column_stack([np.ones(len(y)), x1, x2])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    resid = y - X @ coef
    dof = len(y) - X.shape[1]

    if dof <= 0:
        return coef[1], np.nan, np.nan, coef[2], np.nan, np.nan, len(y)

    s2 = np.sum(resid ** 2) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))

    t = coef / se
    p = 2 * stats.t.sf(np.abs(t), dof)

    return coef[1], t[1], p[1], coef[2], t[2], p[2], len(y)


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

    hits = list(d.glob(f"**/*{sub}*run-{r:02d}*.npy"))
    hits += list(d.glob(f"**/*{sub}*run-{r}*.npy"))

    if hits:
        return sorted(hits)[0]

    return None


def find_bold(sub, run):
    r = int(str(run).replace("run-", ""))

    patterns = [
        f"{sub}_task-lppEN_run-{r:02d}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
        f"{sub}_task-lppEN_run-{r}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
        f"{sub}_*run-{r:02d}*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
        f"{sub}_*run-{r}*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    ]

    for p in patterns:
        hits = sorted((DATA_ROOT / sub / "func").glob(p))
        if hits:
            return hits[0]

    return None


def mean_hipp_signal_from_bold(bold_file, hemi):
    aseg_file = Path(
        str(bold_file).replace(
            "_desc-preproc_bold.nii.gz",
            "_desc-aseg_dseg.nii.gz",
        )
    )

    if not aseg_file.exists():
        raise FileNotFoundError(f"Missing aseg: {aseg_file}")

    bold = nib.load(str(bold_file)).get_fdata()
    aseg = nib.load(str(aseg_file)).get_fdata()

    if bold.shape[:3] != aseg.shape[:3]:
        raise ValueError(f"Shape mismatch for {bold_file}")

    label = HIP_LABELS[hemi]
    mask = aseg == label

    if mask.sum() < 20:
        raise ValueError(f"Too few hippocampal voxels: {bold_file}, hemi={hemi}")

    X = bold[mask, :]
    return np.nanmean(X, axis=0), aseg_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metric",
        choices=list(REGRESSOR_DIRS.keys()) + ["all"],
        default="token_shift",
    )
    ap.add_argument(
        "--trajectory-column",
        default="trajectory_step",
        choices=["trajectory_step", "trajectory_speed_z", "trajectory_angle"],
    )

    args = ap.parse_args()

    metrics = list(REGRESSOR_DIRS.keys()) if args.metric == "all" else [args.metric]
    all_rows = []

    traj_files = sorted(TRAJ_DIR.glob("*_trajectory_features.csv"))
    print(f"Found trajectory files: {len(traj_files)}")

    for metric in metrics:
        print(f"\nMetric: {metric}")

        for traj_file in traj_files:
            traj = pd.read_csv(traj_file)

            if args.trajectory_column not in traj.columns:
                print(f"[skip] missing {args.trajectory_column}: {traj_file}")
                continue

            sub = traj["sub"].iloc[0]
            run = traj["run"].iloc[0]
            hemi = traj["hemi"].iloc[0]

            reg_file = find_regressor(metric, sub, run)
            if reg_file is None:
                print(f"[skip] no regressor: {metric} {sub} {run}")
                continue

            bold_file = find_bold(sub, run)
            if bold_file is None:
                print(f"[skip] no BOLD: {sub} {run}")
                continue

            try:
                mean_signal, aseg_file = mean_hipp_signal_from_bold(bold_file, hemi)
            except Exception as e:
                print(f"[skip] mean signal failed: {sub} {run} {hemi}: {e}")
                continue

            y = traj[args.trajectory_column].values
            x = np.load(reg_file).squeeze().astype(float)
            m = mean_signal

            b_x, t_x, p_x, b_m, t_m, p_m, n = ols_two_predictors(y, x, m)

            all_rows.append({
                "metric": metric,
                "subject": sub,
                "run": f"run-{int(str(run).replace('run-', '')):02d}",
                "hemi": hemi,
                "trajectory_column": args.trajectory_column,
                "beta_metric_control_mean": b_x,
                "t_metric_control_mean": t_x,
                "p_metric_control_mean": p_x,
                "beta_mean_signal": b_m,
                "t_mean_signal": t_m,
                "p_mean_signal": p_m,
                "n_trs": n,
                "trajectory_file": str(traj_file),
                "regressor_file": str(reg_file),
                "bold_file": str(bold_file),
                "aseg_file": str(aseg_file),
            })

            print(f"✅ {metric}: {sub} {run} hemi-{hemi}")

    out = pd.DataFrame(all_rows)

    out_csv = OUT / f"trajectory_{args.trajectory_column}_with_mean_covariate_{args.metric}.csv"
    out.to_csv(out_csv, index=False)

    print(f"\nWrote {out_csv}")
    print(f"Rows: {len(out)}")
    if len(out):
        print(out.head())


if __name__ == "__main__":
    main()