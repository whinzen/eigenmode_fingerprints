#!/usr/bin/env python

import re
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

BASE = Path.home() / "eigenmode_fingerprints"
FUNC_DIR = BASE / "data" / "empirical"

REG_BASE = BASE / "pang_out" / "regressors"
OUT_BASE = BASE / "pang_out" / "vertex_betas"
OUT_BASE.mkdir(parents=True, exist_ok=True)

HEMIS = ["L", "R"]

REGRESSORS = {
    "boundary": REG_BASE / "sentence_boundary_hrf_per_subject",
    "token_shift": REG_BASE / "shift_hrf_per_subject",
    "pred_error_ar": REG_BASE / "pred_error_ar_hrf_per_subject",
    "pred_error_subspace": REG_BASE / "pred_error_subspace_hrf_per_subject",
    "curvature": REG_BASE / "curvature_hrf_per_subject",
}

def parse_run(fname):
    m = re.search(r"_run-(\d+)_", fname)
    if m is None:
        raise ValueError(f"Cannot parse run from {fname}")
    return int(m.group(1))


def load_func_gii(path):
    img = nib.load(str(path))
    data = np.vstack([d.data for d in img.darrays])  # T × V
    return data.astype(float)


def load_regressor(reg_dir, sub, run):
    for fmt in [f"{sub}_run-{run:02d}.npy", f"{sub}_run-{run}.npy"]:
        f = reg_dir / fmt
        if f.exists():
            return np.load(f).astype(float)
    return None


def vertexwise_beta(Y, x):
    good = np.isfinite(x) & np.all(np.isfinite(Y), axis=1)
    Y = Y[good]
    x = x[good]

    if len(x) < 5:
        return None

    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.pinv(X) @ Y
    return beta[1]


def run_glm_for_metric(metric, reg_dir):
    print(f"\n=== Running vertexwise GLM: {metric} ===")

    OUT_DIR = OUT_BASE / metric
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    subjects = sorted([p.name for p in FUNC_DIR.glob("sub-*") if p.is_dir()])

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

                x = load_regressor(reg_dir, sub, run)
                if x is None:
                    continue

                Y = load_func_gii(f)

                T = min(len(x), Y.shape[0])
                beta = vertexwise_beta(Y[:T], x[:T])

                if beta is None:
                    continue

                run_betas.append(beta)

            if run_betas:
                subj_mean = np.nanmean(np.vstack(run_betas), axis=0)
                all_rows.append((sub, hemi, subj_mean))

    # group mean per hemisphere
    for hemi in HEMIS:
        maps = [m for _, h, m in all_rows if h == hemi]
        if not maps:
            continue

        group = np.nanmean(np.vstack(maps), axis=0)

        # save BOTH formats:
        # 1. for record
        np.save(OUT_DIR / f"group_hemi-{hemi}_mean_beta.npy", group)

        # 2. for reconstruction script (simple name)
        np.save(OUT_BASE / f"{metric}_{hemi}.npy", group)

        print(f"✅ {metric} hemi-{hemi}: saved group beta ({group.shape})")


def main():
    for metric, reg_dir in REGRESSORS.items():
        if not reg_dir.exists():
            print(f"[skip] {metric}: missing {reg_dir}")
            continue

        run_glm_for_metric(metric, reg_dir)


if __name__ == "__main__":
    main()