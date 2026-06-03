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

OUT_BASE = PANG / "subcortex" / "hipp_mean_signal_AP"

HIP_LABELS = {
    "L": 17,
    "R": 53,
}

REGRESSOR_DIRS = {
    "sentence_boundary": PANG / "regressors" / "sentence_boundary_hrf_per_subject",
    "sentence_shift": PANG / "regressors" / "sentence_shift_hrf_per_subject",
    "token_shift": PANG / "regressors" / "shift_hrf_per_subject",
    "pred_error_ar": PANG / "regressors" / "pred_error_ar_hrf_per_subject",
    "pred_error_subspace": PANG / "regressors" / "pred_error_subspace_hrf_per_subject",
    "curvature": PANG / "regressors" / "curvature_hrf_per_subject",
}


def infer_run_id(path):
    for part in path.name.split("_"):
        if part.startswith("run-"):
            return part.replace("run-", "")
    raise ValueError(f"Could not infer run from filename: {path}")


def find_regressor(metric, sub, run):
    d = REGRESSOR_DIRS[metric]
    r = int(str(run).replace("run-", ""))

    candidates = [
        d / f"{sub}_run-{r:02d}.npy",
        d / f"{sub}_run-{r}.npy",
        d / f"{sub}_task-lppEN_run-{r:02d}.npy",
        d / f"{sub}_task-lppEN_run-{r}.npy",
    ]

    for f in candidates:
        if f.exists():
            return f

    hits = list(d.glob(f"**/*{sub}*run-{r:02d}*.npy"))
    hits += list(d.glob(f"**/*{sub}*run-{r}*.npy"))

    if hits:
        return sorted(hits)[0]

    return None


def zscore(x):
    x = np.asarray(x, float)
    y = np.full_like(x, np.nan, dtype=float)
    good = np.isfinite(x)

    if good.sum() < 3:
        return y

    sd = np.nanstd(x[good])
    if not np.isfinite(sd) or sd == 0:
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


def split_ap_masks(mask, min_voxels=10):
    """
    Split a hippocampal mask into posterior, middle, anterior thirds
    along voxel y-index.

    In MNI-like orientation, larger y is more anterior.
    Since data are already in MNI152NLin2009cAsym space and on a
    consistent grid, voxel y-index provides a practical AP ordering.
    """
    coords = np.array(np.where(mask)).T

    if len(coords) < 3 * min_voxels:
        return {}

    y = coords[:, 1]
    q1, q2 = np.quantile(y, [1 / 3, 2 / 3])

    parcel_masks = {}

    definitions = {
        "posterior": y <= q1,
        "middle": (y > q1) & (y <= q2),
        "anterior": y > q2,
    }

    for parcel, keep in definitions.items():
        pmask = np.zeros_like(mask, dtype=bool)
        kept_coords = coords[keep]

        if len(kept_coords) < min_voxels:
            continue

        pmask[
            kept_coords[:, 0],
            kept_coords[:, 1],
            kept_coords[:, 2],
        ] = True

        parcel_masks[parcel] = pmask

    return parcel_masks


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

    reg_file = find_regressor(metric, sub, run_raw)

    if reg_file is None:
        print(f"[skip] no regressor: {metric} {sub} run-{run_padded}")
        return []

    bold = nib.load(str(bold_path)).get_fdata()
    aseg = nib.load(str(aseg_path)).get_fdata()

    if bold.ndim != 4:
        print(f"[skip] not 4D BOLD: {bold_path}")
        return []

    if bold.shape[:3] != aseg.shape[:3]:
        print(f"[skip] shape mismatch: {bold_path}")
        return []

    x = np.load(reg_file).squeeze().astype(float)
    rows = []

    for hemi, label in HIP_LABELS.items():
        full_mask = aseg == label
        n_full = int(full_mask.sum())

        if n_full < 30:
            print(f"[skip] {sub} run-{run_padded} hemi-{hemi}: only {n_full} voxels")
            continue

        parcels = split_ap_masks(full_mask, min_voxels=10)

        for parcel, pmask in parcels.items():
            n_voxels = int(pmask.sum())

            X = bold[pmask, :]
            mean_signal = np.nanmean(X, axis=0)

            beta, tval, pval, n_used = ols_beta(mean_signal, x)

            rows.append({
                "subject": sub,
                "sub": sub,
                "run": run_padded,
                "hemi": hemi,
                "parcel": parcel,
                "predictor": metric,
                "metric": metric,
                "n_voxels": n_voxels,
                "n_voxels_full_hemi": n_full,
                "n_trs": min(len(mean_signal), len(x)),
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

    print(f"\nMetric: {metric}")
    print(f"Found BOLD files: {len(bold_files)}")
    print(f"Regressor dir: {REGRESSOR_DIRS[metric]}")

    rows = []

    for bold_path in bold_files:
        rows.extend(process_bold_file(bold_path, metric))

    df = pd.DataFrame(rows)

    nested_csv = out_dir / f"all_{metric}_hipp_mean_signal_AP_glm_rows.csv"
    flat_csv = OUT_BASE / f"mean_signal_AP_{metric}.csv"

    df.to_csv(nested_csv, index=False)
    df.to_csv(flat_csv, index=False)

    print(f"\n✅ wrote {nested_csv}")
    print(f"✅ wrote {flat_csv}")
    print(f"Rows: {len(df)}")

    if len(df) > 0:
        print(df.head())


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--metric",
        choices=list(REGRESSOR_DIRS.keys()) + ["all"],
        default="all",
    )

    args = ap.parse_args()

    metrics = list(REGRESSOR_DIRS.keys()) if args.metric == "all" else [args.metric]

    for metric in metrics:
        run_metric(metric)


if __name__ == "__main__":
    main()