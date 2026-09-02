#!/usr/bin/env python3
"""
standardized_eigenmode_profiles.py

Cortical spectral-envelope control for token-level linguistic eigenmode profiles.

Purpose
-------
The primary eigenmode GLMs estimate effects of linguistic predictors on modal
energy,

    E_k(t) = A_k(t)^2,

where A_k(t) is the cortical eigenmode coefficient at time t.

Because the scale and variance of E_k differ strongly across eigenmodes, raw
OLS beta profiles could in principle inherit part of the generic cortical
modal-energy spectrum.

For a simple OLS model with intercept,

    E_k = b0 + beta_k X + error,

the standardized coefficient is

    beta_std = beta_k * SD(X) / SD(E_k),

which is exactly equal to the Pearson correlation

    r(X, E_k).

This script therefore recomputes token-level effects directly as within-run
Pearson correlations between each HRF-convolved linguistic predictor and each
modal-energy time series.

The resulting profiles remove mode-dependent differences in the scale and
variance of modal energy.

Aggregation follows the primary analyses:

    run / hemisphere
        -> participant mean
        -> group mean

Mode 0 (near-constant mode) is excluded from the reported profiles.

Expected repository layout
--------------------------
~/eigenmode_fingerprints/
    pang_out/
        sub-*/
            run-*/
                A_L.npy
                A_R.npy
        regressors/
            shift_hrf_per_subject/
            pred_error_ar_hrf_per_subject/
            pred_error_subspace_hrf_per_subject/
            curvature_hrf_per_subject/

Outputs
-------
pang_out/standardized_beta_profiles/
    standardized_effects_allruns_token.csv
    standardized_effects_subject_token.csv
    standardized_effects_group_token.csv
    standardized_group_profiles_token.csv
    profile_correlations_token.csv

Author: eigenmode_fingerprints analysis pipeline
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ======================================================================
# Paths
# ======================================================================

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"
REG = PANG / "regressors"

OUT = PANG / "standardized_beta_profiles"
OUT.mkdir(parents=True, exist_ok=True)


# ======================================================================
# Token-level predictors
# ======================================================================

METRICS = {
    "Token shift": REG / "shift_hrf_per_subject",
    "Prediction error": REG / "pred_error_ar_hrf_per_subject",
    "Subspace exit": REG / "pred_error_subspace_hrf_per_subject",
    "Curvature": REG / "curvature_hrf_per_subject",
}


# ======================================================================
# Helpers
# ======================================================================

def corr_modes(E, x):
    """
    Pearson correlation between predictor x [T] and every modal-energy
    time series E [K, T].

    Returns
    -------
    r : ndarray, shape [K]
        Pearson correlation for each eigenmode.
    """

    E = np.asarray(E, dtype=float)
    x = np.asarray(x, dtype=float).squeeze()

    if E.ndim != 2:
        raise ValueError(f"E must be 2D [K,T], got shape {E.shape}")

    if x.ndim != 1:
        raise ValueError(f"x must be 1D [T], got shape {x.shape}")

    if E.shape[1] != len(x):
        raise ValueError(
            f"Time-length mismatch: E has {E.shape[1]} samples, "
            f"x has {len(x)}"
        )

    r = np.full(E.shape[0], np.nan, dtype=float)

    for k in range(E.shape[0]):

        y = E[k]

        good = np.isfinite(x) & np.isfinite(y)

        if good.sum() < 3:
            continue

        xx = x[good]
        yy = y[good]

        xx = xx - xx.mean()
        yy = yy - yy.mean()

        denom = np.sqrt(
            np.sum(xx ** 2) * np.sum(yy ** 2)
        )

        if denom > 0:
            r[k] = np.sum(xx * yy) / denom

    return r


def find_regressor_file(folder, subject, run):
    """
    Return the expected HRF-convolved regressor file.
    """

    path = folder / f"{subject}_run-{run}.npy"

    if path.exists():
        return path

    return None


# ======================================================================
# Check required folders
# ======================================================================

if not PANG.exists():
    raise FileNotFoundError(f"Missing pang_out directory: {PANG}")

for label, folder in METRICS.items():
    if not folder.exists():
        raise FileNotFoundError(
            f"Missing regressor folder for {label}: {folder}"
        )


# ======================================================================
# Subjects
# ======================================================================

subjects = sorted(
    p.name
    for p in PANG.glob("sub-*")
    if p.is_dir()
)

if not subjects:
    raise RuntimeError(
        f"No subject directories found under {PANG}"
    )

print("=" * 72)
print("STANDARDIZED TOKEN-LEVEL EIGENMODE PROFILES")
print("=" * 72)
print(f"Repository: {BASE}")
print(f"Subjects:   {len(subjects)}")
print()


# ======================================================================
# Run-level standardized effects
# ======================================================================

rows = []

usable_runs = 0
skipped_missing_regressor = 0
skipped_missing_modes = 0
skipped_length_mismatch = 0


for subject in subjects:

    subject_dir = PANG / subject

    run_dirs = sorted(
        p for p in subject_dir.glob("run-*")
        if p.is_dir()
    )

    for run_dir in run_dirs:

        try:
            run = int(run_dir.name.split("-")[1])
        except (IndexError, ValueError):
            continue

        # --------------------------------------------------------------
        # Require all four token-level regressors for this run.
        # This keeps the comparison based on exactly the same runs.
        # --------------------------------------------------------------

        regressors = {}
        missing = False

        for label, folder in METRICS.items():

            reg_path = find_regressor_file(
                folder,
                subject,
                run,
            )

            if reg_path is None:
                missing = True
                break

            x = np.asarray(
                np.load(reg_path),
                dtype=float,
            ).squeeze()

            if x.ndim != 1:
                raise ValueError(
                    f"Regressor is not 1D: {reg_path}, shape={x.shape}"
                )

            regressors[label] = x

        if missing:
            skipped_missing_regressor += 1
            continue

        # All regressors should refer to the same TR series.
        lengths = {len(x) for x in regressors.values()}

        if len(lengths) != 1:
            skipped_length_mismatch += 1
            continue

        T = next(iter(lengths))

        # --------------------------------------------------------------
        # Require both hemispheres, matching the primary aggregation.
        # --------------------------------------------------------------

        A_paths = {
            "L": run_dir / "A_L.npy",
            "R": run_dir / "A_R.npy",
        }

        if not all(p.exists() for p in A_paths.values()):
            skipped_missing_modes += 1
            continue

        hemisphere_data = {}

        valid_run = True

        for hemi, A_path in A_paths.items():

            A = np.asarray(
                np.load(A_path),
                dtype=float,
            )

            if A.ndim != 2:
                raise ValueError(
                    f"Modal coefficient array must be 2D: "
                    f"{A_path}, shape={A.shape}"
                )

            if A.shape[1] != T:
                valid_run = False
                break

            hemisphere_data[hemi] = A

        if not valid_run:
            skipped_length_mismatch += 1
            continue

        # --------------------------------------------------------------
        # Compute standardized effects.
        #
        # beta_std = beta * SD(X) / SD(E)
        #          = corr(X, E)
        #
        # where E = A^2.
        # --------------------------------------------------------------

        for hemi, A in hemisphere_data.items():

            E = A ** 2

            for predictor, x in regressors.items():

                r = corr_modes(E, x)

                # Exclude Mode 0 (near-constant mode).
                for mode_k in range(1, len(r)):

                    rows.append(
                        {
                            "predictor": predictor,
                            "subject": subject,
                            "run": run,
                            "hemi": hemi,
                            "mode_k": mode_k,
                            "beta_std": r[mode_k],
                        }
                    )

        usable_runs += 1


# ======================================================================
# Run-level table
# ======================================================================

allruns = pd.DataFrame(rows)

if allruns.empty:
    raise RuntimeError(
        "No usable observations were produced."
    )

print(f"Usable runs: {usable_runs}")
print(f"Run-level rows: {len(allruns):,}")

if skipped_missing_regressor:
    print(
        "Runs skipped for missing predictor(s): "
        f"{skipped_missing_regressor}"
    )

if skipped_missing_modes:
    print(
        "Runs skipped for missing A_L/A_R: "
        f"{skipped_missing_modes}"
    )

if skipped_length_mismatch:
    print(
        "Runs skipped for time-length mismatch: "
        f"{skipped_length_mismatch}"
    )


# ======================================================================
# Participant-level aggregation
#
# Average across runs and hemispheres within each participant.
# ======================================================================

subject_level = (
    allruns
    .groupby(
        [
            "predictor",
            "subject",
            "mode_k",
        ],
        as_index=False,
    )
    .agg(
        beta_std=("beta_std", "mean"),
        N_run_hemi=("beta_std", "count"),
    )
)


# ======================================================================
# Group-level aggregation
# ======================================================================

group_level = (
    subject_level
    .groupby(
        [
            "predictor",
            "mode_k",
        ],
        as_index=False,
    )
    .agg(
        beta_std_mean=("beta_std", "mean"),
        beta_std_sem=(
            "beta_std",
            lambda x:
                x.std(ddof=1)
                / np.sqrt(x.notna().sum())
        ),
        N_subjects=("beta_std", "count"),
    )
)


# ======================================================================
# Wide matrix of group profiles
# ======================================================================

wide = (
    group_level
    .pivot(
        index="mode_k",
        columns="predictor",
        values="beta_std_mean",
    )
    .sort_index()
)


# ======================================================================
# Pairwise profile correlations
# ======================================================================

profile_corr = wide.corr()

print()
print("=" * 72)
print("PAIRWISE CORRELATIONS OF STANDARDIZED GROUP PROFILES")
print("=" * 72)
print(profile_corr.round(6).to_string())


# ======================================================================
# Effect summaries
# ======================================================================

print()
print("=" * 72)
print("STANDARDIZED EFFECT MAGNITUDES")
print("=" * 72)

for predictor in wide.columns:

    x = wide[predictor].dropna()

    print(
        f"{predictor:18s} "
        f"mean={x.mean(): .6f}   "
        f"mean|r|={x.abs().mean(): .6f}   "
        f"min={x.min(): .6f}   "
        f"max={x.max(): .6f}"
    )


# ======================================================================
# First 20 modes
# ======================================================================

print()
print("=" * 72)
print("FIRST 20 MODES")
print("=" * 72)
print(
    wide.head(20)
    .round(5)
    .to_string()
)


# ======================================================================
# Save outputs
# ======================================================================

allruns_path = (
    OUT /
    "standardized_effects_allruns_token.csv"
)

subject_path = (
    OUT /
    "standardized_effects_subject_token.csv"
)

group_path = (
    OUT /
    "standardized_effects_group_token.csv"
)

wide_path = (
    OUT /
    "standardized_group_profiles_token.csv"
)

corr_path = (
    OUT /
    "profile_correlations_token.csv"
)


allruns.to_csv(
    allruns_path,
    index=False,
)

subject_level.to_csv(
    subject_path,
    index=False,
)

group_level.to_csv(
    group_path,
    index=False,
)

wide.to_csv(
    wide_path,
)

profile_corr.to_csv(
    corr_path,
)


# ======================================================================
# Final report
# ======================================================================

print()
print("=" * 72)
print("SAVED")
print("=" * 72)

for path in [
    allruns_path,
    subject_path,
    group_path,
    wide_path,
    corr_path,
]:
    print(path)

print()
print("Done.")