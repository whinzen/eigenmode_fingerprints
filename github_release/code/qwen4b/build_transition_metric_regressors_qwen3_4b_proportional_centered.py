#!/usr/bin/env python

import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.special import gamma
from tqdm import tqdm


# ============================================================
# Paths / config
# ============================================================

BASE = Path.home() / "eigenmode_fingerprints"

TRANS_CSV = (
    BASE
    / "pang_out"
    / "word_transition_geometry_qwen3_4b_proportional"
    / "word_transition_geometry.csv"
)

ANNOT_CSV = (
    BASE
    / "ds003643"
    / "annotation"
    / "EN"
    / "repunct"
    / "lppEN.csv"
)

FUNC_DIR = BASE / "data" / "empirical"

OUT_BASE = (
    BASE
    / "pang_out"
    / "regressors_qwen3_4b_proportional_centered"
)

OUT_BASE.mkdir(parents=True, exist_ok=True)

TR = 2.0

METRICS = [
    "shift",
    "pred_error_ar",
    "pred_error_subspace",
    "curvature",
]


# ============================================================
# Helpers
# ============================================================

def get_n_trs_from_gifti(func_file: Path) -> int:
    img = nib.load(str(func_file))
    return len(img.darrays)


def canonical_hrf(tr=2.0, duration=32.0):
    """
    Same simple SPM-like double-gamma HRF as original Qwen pipeline.
    Sampled at TR resolution.
    """
    t = np.arange(0, duration, tr)

    def gamma_pdf(x, a, scale=1.0):
        x = np.asarray(x, float)
        out = np.zeros_like(x)

        m = x > 0
        xm = x[m]

        out[m] = (
            (xm ** (a - 1))
            * np.exp(-xm / scale)
            / (gamma(a) * (scale ** a))
        )

        return out

    h = gamma_pdf(t, 6) - (1 / 6) * gamma_pdf(t, 16)

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
        return np.zeros_like(x)

    return (x - mu) / sd


def parse_run_number(path: Path) -> int:
    m = re.search(r"_run-(\d+)_", path.name)

    if not m:
        raise ValueError(
            f"Could not parse run number from {path.name}"
        )

    return int(m.group(1))


def get_subject_run_mapping(func_subj_dir: Path):
    """
    Annotation run 1 -> first actual available fMRI run
    Annotation run 2 -> second actual available fMRI run
    etc.
    """

    run_files = sorted(
        func_subj_dir.glob(
            "*_hemi-L_space-fsaverage5_bold.func.gii"
        )
    )

    if len(run_files) == 0:
        return {}

    actual_runs = sorted(
        set(parse_run_number(p) for p in run_files)
    )

    return {
        ann_run: actual_run
        for ann_run, actual_run
        in enumerate(actual_runs, start=1)
    }


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


# ============================================================
# Main
# ============================================================

def main():

    if not TRANS_CSV.exists():
        raise FileNotFoundError(
            f"Missing transition table: {TRANS_CSV}"
        )

    if not ANNOT_CSV.exists():
        raise FileNotFoundError(
            f"Missing annotation table: {ANNOT_CSV}"
        )

    if not FUNC_DIR.exists():
        raise FileNotFoundError(
            f"Missing functional data directory: {FUNC_DIR}"
        )

    trans = pd.read_csv(TRANS_CSV)
    annot = pd.read_csv(ANNOT_CSV)

    # --------------------------------------------------------
    # Continuous-Qwen transition table
    # --------------------------------------------------------

    required = {
        "run_id",
        "onset",
        "boundary_flag",
        *METRICS,
    }

    missing = required.difference(trans.columns)

    if missing:
        raise RuntimeError(
            f"Missing transition columns: {sorted(missing)}"
        )

    if "same_run" in trans.columns:
        trans = trans[trans["same_run"] == 1].copy()

    trans["run_id"] = trans["run_id"].astype(int)

    # Center every metric WITHIN annotation run
    for metric in METRICS:
        trans[f"{metric}_centered"] = (
            trans[metric]
            - trans.groupby("run_id")[metric].transform("mean")
        )

    # --------------------------------------------------------
    # Original annotation for complete word-rate control
    # --------------------------------------------------------

    run_col = "run_id"

    onset_candidates = [
        "onset",
        "onset_sec",
        "word_onset",
        "start",
    ]

    onset_col = next(
        (c for c in onset_candidates if c in annot.columns),
        None,
    )

    if onset_col is None:
        raise RuntimeError(
            f"Could not identify annotation onset column. "
            f"Columns: {list(annot.columns)}"
        )

    if run_col not in annot.columns:
        raise RuntimeError(
            f"Annotation missing {run_col}"
        )

    annot[run_col] = annot[run_col].astype(int)

    print("Continuous transition table:", TRANS_CSV)
    print("Transition rows:", len(trans))
    print("Annotation rows:", len(annot))
    print("Annotation onset column:", onset_col)
    print("Output root:", OUT_BASE)

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    predictors = ["word_rate"] + [
        f"{m}_centered" for m in METRICS
    ]

    for predictor in predictors:
        d = OUT_BASE / f"{predictor}_hrf_per_subject"
        d.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Subjects
    # --------------------------------------------------------

    subjects = sorted([
        p.name
        for p in FUNC_DIR.glob("sub-*")
        if p.is_dir()
    ])

    print("Found subjects:", len(subjects))

    n_saved = 0

    for subj in tqdm(subjects, desc="Subjects"):

        func_subj_dir = FUNC_DIR / subj / "func"

        if not func_subj_dir.exists():
            continue

        mapping = get_subject_run_mapping(func_subj_dir)

        if not mapping:
            print(
                f"[skip] {subj}: no fsaverage5 L hemi files"
            )
            continue

        for ann_run_id, actual_run in mapping.items():

            func_file = find_left_func_file(
                func_subj_dir,
                actual_run,
            )

            if func_file is None:
                print(
                    f"[skip] {subj} run-{actual_run}: "
                    f"no left func file"
                )
                continue

            n_trs = get_n_trs_from_gifti(func_file)

            # ================================================
            # A. Complete unweighted word-rate regressor
            # ================================================

            ann_run = annot[
                annot[run_col] == ann_run_id
            ].copy()

            word_onsets = ann_run[onset_col].values

            word_rate = build_impulse_regressor(
                word_onsets,
                np.ones(len(word_onsets)),
                n_trs=n_trs,
                tr=TR,
            )

            word_rate = convolve_hrf(
                word_rate,
                HRF,
            )

            word_rate = zscore_safe(word_rate)

            out_path = (
                OUT_BASE
                / "word_rate_hrf_per_subject"
                / f"{subj}_run-{actual_run}.npy"
            )

            np.save(out_path, word_rate)
            n_saved += 1

            # ================================================
            # B. Centered continuous-Qwen geometry regressors
            # ================================================

            run_df = trans[
                trans["run_id"] == ann_run_id
            ].copy()

            if run_df.empty:
                continue

            onsets = run_df["onset"].values

            for metric in METRICS:

                col = f"{metric}_centered"

                vals = run_df[col].values

                good = (
                    np.isfinite(onsets)
                    & np.isfinite(vals)
                )

                if good.sum() == 0:

                    x = np.zeros(
                        n_trs,
                        dtype=float,
                    )

                else:

                    x = build_impulse_regressor(
                        onsets[good],
                        vals[good],
                        n_trs=n_trs,
                        tr=TR,
                    )

                    x = convolve_hrf(
                        x,
                        HRF,
                    )

                    x = zscore_safe(x)

                out_path = (
                    OUT_BASE
                    / f"{col}_hrf_per_subject"
                    / f"{subj}_run-{actual_run}.npy"
                )

                np.save(out_path, x)
                n_saved += 1

    print(
        "\n✅ Done building revised continuous-Qwen regressors."
    )

    print(
        f"Saved {n_saved} files under {OUT_BASE}"
    )


if __name__ == "__main__":
    main()