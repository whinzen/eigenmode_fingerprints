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
OUT_BASE = PANG / "subcortex" / "hipp_mean_signal"

REGRESSOR_DIRS = {
    "token_shift": PANG / "regressors" / "shift_hrf_per_subject",
    "pred_error_ar": PANG / "regressors" / "pred_error_ar_hrf_per_subject",
    "pred_error_subspace": PANG / "regressors" / "pred_error_subspace_hrf_per_subject",
    "curvature": PANG / "regressors" / "curvature_hrf_per_subject",
}

HIP_LABELS = {
    "L": 17,
    "R": 53,
}


def infer_run_id(path):
    for part in path.name.split("_"):
        if part.startswith("run-"):
            return part.replace("run-", "")
    raise ValueError(f"Could not infer run from filename: {path}")


def find_regressor(metric, sub, run):
    d = REGRESSOR_DIRS[metric]

    run_num = str(int(str(run).replace("run-", "")))
    run_num_2 = f"{int(run_num):02d}"

    candidates = [
        d / f"{sub}_run-{run_num_2}.npy",
        d / f"{sub}_run-{run_num}.npy",
        d / f"{sub}_task-lppEN_run-{run_num_2}.npy",
        d / f"{sub}_task-lppEN_run-{run_num}.npy",
    ]

    for f in candidates:
        if f.exists():
            return f

    hits = list(d.glob(f"**/*{sub}*run-{run_num_2}*.npy"))
    hits += list(d.glob(f"**/*{sub}*run-{run_num}*.npy"))

    if hits:
        return sorted(hits)[0]

    raise FileNotFoundError(f"No regressor found for {metric}: {sub} run-{run_num_2}")


def zscore(x):
    x = np.asarray(x, float)
    good = np.isfinite(x)
    y = np.full_like(x, np.nan, dtype=float)

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

    if len(y) < 5 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, np.nan, np.nan, len(y)

    X = np.column_stack([np.ones(len(x)), x])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    resid = y - X @ coef
    dof = len(y) - X.shape[1]

    if dof <= 0:
        return coef[1], np.nan, np.nan, len(y)

    s2 = np.sum(resid ** 2) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(cov[1, 1])

    if se == 0 or not np.isfinite(se):
        return coef[1], np.nan, np.nan, len(y)

    tval = coef[1] / se
    pval = 2 * stats.t.sf(np.abs(tval), dof)

    return coef[1], tval, pval, len(y)


def process_bold_file(bold_path, metric):
    sub = bold_path.parts[-3]
    run_raw = infer_run_id(bold_path)
    run_padded = f"{int(run_raw):02d}"

    aseg_path = Path(
        str(bold_path).replace(
            "_desc-preproc_bold.nii.gz",
            "_desc-aseg_dseg.nii.gz",
        )
    )

    if not aseg_path.exists():
        print(f"[skip] missing aseg: {aseg_path}")
        return []

    try:
        reg_file = find_regressor(metric, sub, run_raw)
    except FileNotFoundError as e:
        print(f"[skip] {e}")
        return []

    bold_img = nib.load(str(bold_path))
    aseg_img = nib.load(str(aseg_path))

    bold = bold_img.get_fdata()
    aseg = aseg_img.get_fdata()

    if bold.ndim != 4:
        print(f"[skip] not 4D BOLD: {bold_path}")
        return []

    if bold.shape[:3] != aseg.shape[:3]:
        print(f"[skip] shape mismatch: {bold_path}")
        return []

    x = np.load(reg_file).squeeze().astype(float)

    rows = []

    for hemi, label in HIP_LABELS.items():
        mask = aseg == label
        n_voxels = int(mask.sum())

        if n_voxels < 20:
            print(f"[skip] {sub} run-{run_padded} hemi-{hemi}: only {n_voxels} voxels")
            continue

        # voxel x time
        Xbold = bold[mask, :]
        mean_signal = np.nanmean(Xbold, axis=0)

        n = min(len(mean_signal), len(x))

        y = zscore(mean_signal[:n])
        xz = zscore(x[:n])

        beta, tval, pval, n_used = ols_beta(y, xz)

        rows.append({
            "subject": sub,
            "sub": sub,
            "run": run_padded,
            "hemi": hemi,
            "predictor": metric,
            "metric": metric,
            "n_voxels": n_voxels,
            "n_trs": n,
            "n_used": n_used,
            "beta": beta,
            "t": tval,
            "p": pval,
            "regressor_file": str(reg_file),
            "bold_file": str(bold_path),
            "aseg_file": str(aseg_path),
        })

    print(f"✅ {metric}: {sub} run-{run_padded}")

    return rows


def run_metric(metric):
    out_dir = OUT_BASE / metric
    out_dir.mkdir(parents=True, exist_ok=True)

    bold_files = sorted(
        DATA_ROOT.glob(
            "sub-EN*/func/*task-lppEN*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        )
    )

    rows = []

    print(f"\nMetric: {metric}")
    print(f"Found BOLD files: {len(bold_files)}")
    print(f"Regressor dir: {REGRESSOR_DIRS[metric]}")

    for bold_path in bold_files:
        rows.extend(process_bold_file(bold_path, metric))

    all_df = pd.DataFrame(rows)

    all_csv = out_dir / f"all_{metric}_hipp_mean_signal_glm_rows.csv"
    all_df.to_csv(all_csv, index=False)

    flat_csv = OUT_BASE / f"mean_signal_{metric}.csv"
    all_df.to_csv(flat_csv, index=False)

    print(f"\n✅ wrote {all_csv}")
    print(f"✅ wrote {flat_csv}")
    print(f"Rows: {len(all_df)}")

    if len(all_df) > 0:
        print(all_df.head())


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--metric",
        choices=[
            "token_shift",
            "pred_error_ar",
            "pred_error_subspace",
            "curvature",
            "all",
        ],
        default="all",
    )

    args = ap.parse_args()

    metrics = list(REGRESSOR_DIRS.keys()) if args.metric == "all" else [args.metric]

    for metric in metrics:
        run_metric(metric)


if __name__ == "__main__":
    main()