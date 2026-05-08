#!/usr/bin/env python

import re
import math
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from tqdm import tqdm


# =========================
# Paths / config
# =========================
BASE = Path.home() / "eigenmode_fingerprints"

TRANS_CSV = (
    BASE
    / "pang_out"
    / "word_transition_geometry_qwen3_0p6b"
    / "word_transition_geometry.csv"
)

# Subject fsaverage5 BOLD files
FUNC_DIR = BASE / "data" / "empirical"

# Qwen-specific output root
OUT_BASE = BASE / "pang_out" / "regressors_qwen3_0p6b"
OUT_BASE.mkdir(parents=True, exist_ok=True)

TR = 2.0

METRICS = [
    "shift",
    "pred_error_ar",
    "pred_error_subspace",
    "curvature",
]


# =========================
# Helpers
# =========================
def get_n_trs_from_gifti(func_file: Path) -> int:
    img = nib.load(str(func_file))
    return len(img.darrays)


def canonical_hrf(tr=2.0, duration=32.0):
    """
    Simple SPM-like double-gamma HRF sampled at TR.
    """
    t = np.arange(0, duration, tr)

    def gamma_pdf(x, a, scale=1.0):
        x = np.asarray(x, float)
        out = np.zeros_like(x)
        m = x > 0
        xm = x[m]
        out[m] = (xm ** (a - 1) * np.exp(-xm / scale)) / (
            math.gamma(a) * scale ** a
        )
        return out

    h = gamma_pdf(t, 6) - 0.5 * gamma_pdf(t, 16)
    if np.max(np.abs(h)) > 0:
        h = h / np.max(np.abs(h))
    return h


HRF = canonical_hrf(TR)


def convolve_hrf(x, hrf):
    return np.convolve(x, hrf, mode="full")[: len(x)]


def zscore_safe(x):
    x = np.asarray(x, float)
    good = np.isfinite(x)

    if good.sum() == 0:
        return np.zeros_like(x)

    mu = np.nanmean(x)
    sd = np.nanstd(x)

    if not np.isfinite(sd) or sd == 0:
        return x - mu

    return (x - mu) / sd


def parse_run_number(path: Path) -> int:
    m = re.search(r"_run-(\d+)_", path.name)
    if not m:
        raise ValueError(f"Could not parse run number from {path.name}")
    return int(m.group(1))


def get_subject_run_mapping(func_subj_dir: Path):
    """
    Map annotation run ids 1..N to actual subject run numbers by sorting subject runs.

    Example:
    annotation run 1 -> first actual available run number for this subject
    annotation run 2 -> second actual available run number, etc.
    """
    run_files = sorted(
        func_subj_dir.glob("*_hemi-L_space-fsaverage5_bold.func.gii")
    )

    if len(run_files) == 0:
        return {}

    actual_runs = sorted(parse_run_number(p) for p in run_files)
    return {i + 1: run for i, run in enumerate(actual_runs)}


def find_left_func_file(func_subj_dir: Path, actual_run: int):
    candidates = sorted(
        func_subj_dir.glob(
            f"*_run-{actual_run}_hemi-L_space-fsaverage5_bold.func.gii"
        )
    )

    if len(candidates) == 0:
        candidates = sorted(
            func_subj_dir.glob(
                f"*_run-{actual_run:02d}_hemi-L_space-fsaverage5_bold.func.gii"
            )
        )

    return candidates[0] if candidates else None


def build_impulse_regressor(onsets_sec, values, n_trs, tr=2.0):
    x = np.zeros(n_trs, dtype=float)

    for onset, val in zip(onsets_sec, values):
        if not np.isfinite(onset) or not np.isfinite(val):
            continue

        idx = int(round(float(onset) / tr))

        if 0 <= idx < n_trs:
            x[idx] += float(val)

    return x


# =========================
# Main
# =========================
def main():
    if not TRANS_CSV.exists():
        raise FileNotFoundError(f"Missing transition table: {TRANS_CSV}")

    if not FUNC_DIR.exists():
        raise FileNotFoundError(f"Missing functional data directory: {FUNC_DIR}")

    trans = pd.read_csv(TRANS_CSV)

    required = {"run_id", "onset", "boundary_flag", *METRICS}
    missing = required.difference(trans.columns)

    if missing:
        raise RuntimeError(f"Missing columns in transition table: {sorted(missing)}")

    if "same_run" in trans.columns:
        trans = trans[trans["same_run"] == 1].copy()

    trans["run_id"] = trans["run_id"].astype(int)

    print(f"Loaded transition table: {TRANS_CSV}")
    print(f"Transition rows: {len(trans)}")
    print(f"Output root: {OUT_BASE}")

    for metric in METRICS:
        d = OUT_BASE / f"{metric}_hrf_per_subject"
        d.mkdir(parents=True, exist_ok=True)

    subjects = sorted([p.name for p in FUNC_DIR.glob("sub-*") if p.is_dir()])
    print(f"Found {len(subjects)} subjects")

    n_saved = 0

    for subj in tqdm(subjects, desc="Subjects"):
        func_subj_dir = FUNC_DIR / subj / "func"

        if not func_subj_dir.exists():
            continue

        mapping = get_subject_run_mapping(func_subj_dir)

        if not mapping:
            print(f"[skip] {subj}: no fsaverage5 L hemi files")
            continue

        for ann_run_id, actual_run in mapping.items():
            func_file = find_left_func_file(func_subj_dir, actual_run)

            if func_file is None:
                print(f"[skip] {subj} run-{actual_run}: no left func file")
                continue

            n_trs = get_n_trs_from_gifti(func_file)

            run_df = trans[trans["run_id"] == ann_run_id].copy()

            if run_df.empty:
                continue

            onsets = run_df["onset"].values

            for metric in METRICS:
                vals = run_df[metric].values
                good = np.isfinite(onsets) & np.isfinite(vals)

                if good.sum() == 0:
                    x = np.zeros(n_trs, dtype=float)
                else:
                    x = build_impulse_regressor(
                        onsets[good],
                        vals[good],
                        n_trs=n_trs,
                        tr=TR,
                    )
                    x = convolve_hrf(x, HRF)
                    x = zscore_safe(x)

                out_path = (
                    OUT_BASE
                    / f"{metric}_hrf_per_subject"
                    / f"{subj}_run-{actual_run}.npy"
                )

                np.save(out_path, x)
                n_saved += 1

    print(f"✅ Done building Qwen transition-metric regressors.")
    print(f"Saved {n_saved} files under {OUT_BASE}")


if __name__ == "__main__":
    main()