#!/usr/bin/env python

import re
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

BASE = Path.home() / "eigenmode_fingerprints"
FUNC_DIR = BASE / "data" / "empirical"
REG_DIR = BASE / "pang_out" / "regressors" / "sentence_shift_hrf_per_subject"

OUT_DIR = BASE / "pang_out" / "vertexwise_betas" / "sentence_shift"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEMIS = ["L", "R"]


def parse_run(fname):
    m = re.search(r"_run-(\d+)_", fname)
    if m is None:
        raise ValueError(f"Could not parse run from {fname}")
    return int(m.group(1))


def load_func_gii(path):
    img = nib.load(str(path))
    # darrays are timepoints, each shape (vertices,)
    data = np.vstack([d.data for d in img.darrays])  # T × V
    return data.astype(float)


def load_regressor(sub, run):
    candidates = [
        REG_DIR / f"{sub}_run-{run:02d}.npy",
        REG_DIR / f"{sub}_run-{run}.npy",
    ]
    for f in candidates:
        if f.exists():
            return np.load(f).astype(float)
    return None


def vertexwise_beta(Y, x):
    """
    Y: T × V
    x: T

    Fits Y_v(t) = b0_v + b1_v * x(t) + error.
    Returns beta map b1, shape V.
    """
    good = np.isfinite(x) & np.all(np.isfinite(Y), axis=1)
    Y = Y[good]
    x = x[good]

    if len(x) < 5:
        return None

    X = np.column_stack([np.ones(len(x)), x])

    # beta = (X'X)^-1 X'Y
    beta = np.linalg.pinv(X) @ Y
    return beta[1]


def main():
    subjects = sorted([p.name for p in FUNC_DIR.glob("sub-*") if p.is_dir()])
    print(f"Found {len(subjects)} subjects")

    all_rows = []

    for sub in subjects:
        func_subdir = FUNC_DIR / sub / "func"
        if not func_subdir.exists():
            continue

        for hemi in HEMIS:
            files = sorted(
                func_subdir.glob(
                    f"{sub}_task-lppEN_run-*_hemi-{hemi}_space-fsaverage5_bold.func.gii"
                )
            )

            run_betas = []

            for f in tqdm(files, desc=f"{sub} hemi-{hemi}"):
                run = parse_run(f.name)
                x = load_regressor(sub, run)

                if x is None:
                    print(f"[skip] {sub} run-{run} hemi-{hemi}: missing regressor")
                    continue

                Y = load_func_gii(f)  # T × V

                T = min(len(x), Y.shape[0])
                beta = vertexwise_beta(Y[:T], x[:T])

                if beta is None:
                    print(f"[skip] {sub} run-{run} hemi-{hemi}: beta failed")
                    continue

                run_betas.append(beta)

                out_run = OUT_DIR / f"{sub}_run-{run:02d}_hemi-{hemi}_beta.npy"
                np.save(out_run, beta)

            if run_betas:
                subj_mean = np.nanmean(np.vstack(run_betas), axis=0)
                out_sub = OUT_DIR / f"{sub}_hemi-{hemi}_mean_beta.npy"
                np.save(out_sub, subj_mean)

                all_rows.append((sub, hemi, subj_mean))

    # group mean per hemi
    for hemi in HEMIS:
        maps = [m for _, h, m in all_rows if h == hemi]
        if not maps:
            continue

        group = np.nanmean(np.vstack(maps), axis=0)
        out_group = OUT_DIR / f"group_hemi-{hemi}_mean_beta.npy"
        np.save(out_group, group)

        print(f"✅ wrote {out_group} shape={group.shape}")


if __name__ == "__main__":
    main()