#!/usr/bin/env python
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import zscore


HIP_LABELS = {
    "L": 17,
    "R": 53,
}


def infer_run_id(path):
    name = path.name

    for part in name.split("_"):
        if part.startswith("run-"):
            return part.replace("run-", "")

    raise ValueError(f"Could not infer run from filename: {path}")


def load_regressor(path):
    x = np.load(path)
    x = np.asarray(x).squeeze()

    if x.ndim != 1:
        raise ValueError(f"Regressor is not 1D: {path}, shape={x.shape}")

    return x


def find_regressor(reg_root, pattern, subject, run, predictor):
    run_int = int(run)
    run_unpadded = str(run_int)
    run_padded = f"{run_int:02d}"

    candidate_names = []

    for r in [run_padded, run_unpadded]:
        try:
            candidate_names.append(
                pattern.format(
                    subject=subject,
                    run=r,
                    run_padded=run_padded,
                    run_unpadded=run_unpadded,
                    predictor=predictor,
                )
            )
        except KeyError:
            pass

    candidate_names.extend([
        f"{subject}_run-{run_padded}.npy",
        f"{subject}_run-{run_unpadded}.npy",
    ])

    # remove duplicates while preserving order
    candidate_names = list(dict.fromkeys(candidate_names))

    for name in candidate_names:
        path = reg_root / name
        if path.exists():
            return path, candidate_names

    return None, candidate_names


def fit_ols(y, x):
    n = min(len(y), len(x))
    y = y[:n]
    x = x[:n]

    ok = np.isfinite(y) & np.isfinite(x)
    y = y[ok]
    x = x[ok]

    if len(y) < 20 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, np.nan, len(y)

    yz = zscore(y)
    xz = zscore(x)

    design = sm.add_constant(xz)
    model = sm.OLS(yz, design).fit()

    return model.params[1], model.tvalues[1], model.pvalues[1], len(yz)


def process_run(bold_path, aseg_path, reg_path, subject, run, predictor):
    bold_img = nib.load(str(bold_path))
    aseg_img = nib.load(str(aseg_path))

    bold = bold_img.get_fdata()
    aseg = aseg_img.get_fdata()

    if bold.ndim != 4:
        raise ValueError(f"BOLD image is not 4D: {bold_path}")

    if bold.shape[:3] != aseg.shape[:3]:
        raise ValueError(
            f"Shape mismatch: BOLD {bold.shape[:3]} vs aseg {aseg.shape[:3]}"
        )

    reg = load_regressor(reg_path)

    rows = []

    for hemi, label in HIP_LABELS.items():
        mask = aseg == label
        n_voxels = int(mask.sum())

        if n_voxels < 20:
            print(f"[skip] {subject} run-{run} hemi-{hemi}: only {n_voxels} voxels")
            continue

        # voxel x time
        X = bold[mask, :]

        # mean hippocampal BOLD signal per TR
        mean_signal = np.nanmean(X, axis=0)

        beta, tval, pval, n_tp = fit_ols(mean_signal, reg)

        rows.append({
            "subject": subject,
            "run": run,
            "hemi": hemi,
            "predictor": predictor,
            "n_voxels": n_voxels,
            "n_tp": n_tp,
            "beta": beta,
            "t": tval,
            "p": pval,
            "bold_path": str(bold_path),
            "aseg_path": str(aseg_path),
            "regressor_path": str(reg_path),
        })

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Run hippocampal mean-signal GLMs with padded/unpadded run support."
    )

    parser.add_argument(
        "--data-root",
        required=True,
        help="Path to data/empirical",
    )

    parser.add_argument(
        "--regressor-root",
        required=True,
        help="Directory containing subject/run HRF regressors",
    )

    parser.add_argument(
        "--predictor",
        required=True,
        help="Predictor label to write into output CSV",
    )

    parser.add_argument(
        "--regressor-pattern",
        default="{subject}_run-{run_padded}.npy",
        help=(
            "Filename pattern. Available fields: "
            "{subject}, {run}, {run_padded}, {run_unpadded}, {predictor}"
        ),
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output CSV path",
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    reg_root = Path(args.regressor_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bold_paths = sorted(
        data_root.glob(
            "sub-EN*/func/*task-lppEN*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        )
    )

    print(f"Found {len(bold_paths)} BOLD files")
    print(f"Regressor root: {reg_root}")
    print(f"Regressor pattern: {args.regressor_pattern}")

    all_rows = []
    n_missing_regressors = 0
    n_missing_aseg = 0
    n_errors = 0

    for bold_path in bold_paths:
        subject = bold_path.parts[-3]
        run = infer_run_id(bold_path)
        run_padded = f"{int(run):02d}"

        aseg_path = Path(
            str(bold_path).replace(
                "_desc-preproc_bold.nii.gz",
                "_desc-aseg_dseg.nii.gz",
            )
        )

        if not aseg_path.exists():
            print(f"[skip] missing aseg: {aseg_path}")
            n_missing_aseg += 1
            continue

        reg_path, tried = find_regressor(
            reg_root=reg_root,
            pattern=args.regressor_pattern,
            subject=subject,
            run=run,
            predictor=args.predictor,
        )

        if reg_path is None:
            print(
                f"[skip] missing regressor for {subject} run-{run_padded}; "
                f"tried: {tried}"
            )
            n_missing_regressors += 1
            continue

        print(f"[run] {subject} run-{run_padded} {args.predictor}")

        try:
            rows = process_run(
                bold_path=bold_path,
                aseg_path=aseg_path,
                reg_path=reg_path,
                subject=subject,
                run=run_padded,
                predictor=args.predictor,
            )
            all_rows.extend(rows)

        except Exception as e:
            print(f"[error] {bold_path}: {e}")
            n_errors += 1

    df = pd.DataFrame(all_rows)
    df.to_csv(out_path, index=False)

    print("\nDone.")
    print(f"Saved: {out_path}")
    print(f"Rows: {len(df)}")
    print(f"Missing regressors: {n_missing_regressors}")
    print(f"Missing aseg files: {n_missing_aseg}")
    print(f"Errors: {n_errors}")

    if len(df) > 0:
        print(df.head())


if __name__ == "__main__":
    main()