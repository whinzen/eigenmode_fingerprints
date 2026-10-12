# ============================================================
# SIMPLE UPSTREAM-OF-HRF TOKEN TEMPORAL NULL
#
# Five INDEPENDENT predictor-specific nulls.
#
# For each predictor and narrative run:
#
#   native finite token values
#       -> circularly rotate values across that predictor's
#          ORIGINAL eligible token positions
#       -> weighted impulses at ORIGINAL onsets
#       -> HRF convolution
#       -> z-score
#       -> GLM: E_k ~ 1 + word_rate + shifted predictor
#
# Different predictor = independently sampled lag.
#
# Preserves:
#   - exact empirical eligible event set for each predictor
#   - token-value distribution
#   - sequential ordering/autocorrelation under circular shift
#   - actual word/event timing
#   - HRF construction
#   - correctly aligned word-rate nuisance
#
# Tests:
#   Is the correctly aligned predictor effect unusually large
#   relative to temporally displaced versions of THAT predictor?
#
# IMPORTANT:
#   First run with N_PERM = 10.
#   Only increase to 10000 after both empirical QC checks pass.
# ============================================================

from pathlib import Path
import re
import time
import os

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.special import gamma


# ============================================================
# SETTINGS
# ============================================================

N_PERM = int(os.environ.get("N_PERM", "10"))  # QC default; set N_PERM=10000 for final run
SEED = 20260915

MIN_TRS = 100

# Avoid nearly unshifted sequences.
# A circular shift must move at least 10% of the token sequence
# away from either zero shift or a complete wraparound.
MIN_SHIFT_FRAC = 0.10


# ============================================================
# PATHS
# ============================================================

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

TRANS_CSV = (
    PANG
    / "word_transition_geometry_qwen3_4b_proportional"
    / "word_transition_geometry.csv"
)

SURP_CSV = (
    PANG
    / "token_surprisal_qwen3_4b_proportional"
    / "token_surprisal_qwen3_4b_continuous.csv"
)

FUNC_DIR = (
    BASE
    / "data"
    / "empirical"
)

REG_ROOT = (
    PANG
    / "regressors_qwen3_4b_proportional_centered"
)

# Direct modal-data root.
# Expected structure:
#
# pang_out/
#   sub-ENxxx/
#       run-X/
#           A_L.npy
#           A_R.npy
#
MODAL_ROOT = PANG / "nuisance_robustness" / "corrected_modal_amplitudes"


# Saved standardized profiles used in current token figure
TOKEN_FILE = (
    PANG
    / "standardized_beta_profiles_qwen3_4b_proportional_centered_wordrate"
    / "standardized_group_profiles_token.csv"
)

SURP_FILE = (
    PANG
    / "standardized_beta_profiles_qwen3_4b_proportional_centered_wordrate_surprisal"
    / "standardized_group_profile_surprisal.csv"
)


# ============================================================
# OUTPUT
# ============================================================

OUT_ROOT = (
    PANG
    / "nuisance_robustness"
    / "figure3_corrected"
    / "token_upstream_hrf_temporal_null_qwen3_4b_proportional"
)

OUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EXACT HRF / EVENT FUNCTIONS FROM EMPIRICAL BUILDER
# ============================================================

TR = 2.0


def canonical_hrf(tr=2.0, duration=32.0):

    t = np.arange(
        0,
        duration,
        tr
    )

    def gamma_pdf(
        x,
        a,
        scale=1.0
    ):

        x = np.asarray(
            x,
            float
        )

        out = np.zeros_like(x)

        m = x > 0
        xm = x[m]

        out[m] = (
            (xm ** (a - 1))
            * np.exp(-xm / scale)
            / (
                gamma(a)
                * (scale ** a)
            )
        )

        return out

    h = (
        gamma_pdf(t, 6)
        -
        (1 / 6)
        * gamma_pdf(t, 16)
    )

    if np.max(
        np.abs(h)
    ) > 0:

        h = (
            h
            / np.max(
                np.abs(h)
            )
        )

    return h


HRF = canonical_hrf(TR)


def convolve_hrf(x, hrf):

    return np.convolve(
        x,
        hrf,
        mode="full"
    )[:len(x)]


def zscore_safe(x):

    x = np.asarray(
        x,
        float
    )

    good = np.isfinite(x)

    if good.sum() == 0:
        return np.zeros_like(x)

    mu = np.nanmean(x)
    sd = np.nanstd(x)

    if (
        not np.isfinite(sd)
        or sd == 0
    ):
        return np.zeros_like(x)

    return (
        x - mu
    ) / sd


def build_impulse_regressor(
    onsets_sec,
    values,
    n_trs,
    tr=2.0
):

    x = np.zeros(
        n_trs,
        dtype=float
    )

    for onset, val in zip(
        onsets_sec,
        values
    ):

        if (
            not np.isfinite(onset)
            or
            not np.isfinite(val)
        ):
            continue

        idx = int(
            round(
                float(onset)
                / tr
            )
        )

        if (
            0 <= idx < n_trs
        ):
            x[idx] += float(val)

    return x


# ============================================================
# fMRI RUN HELPERS
# ============================================================

def parse_run_number(path):

    m = re.search(
        r"_run-(\d+)_",
        path.name
    )

    if not m:
        raise ValueError(
            f"Could not parse run number from {path.name}"
        )

    return int(
        m.group(1)
    )


def get_subject_run_mapping(
    func_subj_dir
):

    run_files = sorted(
        func_subj_dir.glob(
            "*_hemi-L_space-fsaverage5_bold.func.gii"
        )
    )

    if len(run_files) == 0:
        return {}

    actual_runs = sorted(
        set(
            parse_run_number(p)
            for p in run_files
        )
    )

    return {
        ann_run: actual_run
        for ann_run, actual_run
        in enumerate(
            actual_runs,
            start=1
        )
    }


def find_left_func_file(
    func_subj_dir,
    actual_run
):

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

    return (
        candidates[0]
        if candidates
        else None
    )


def get_n_trs(
    func_file
):

    return len(
        nib.load(
            str(func_file)
        ).darrays
    )


def find_confounds_file(func_dir, run):
    """Locate the fMRIPrep confounds TSV for an actual scanner run."""
    pats = [
        f"*run-{run}_desc-confounds_timeseries.tsv",
        f"*run-{run:02d}_desc-confounds_timeseries.tsv",
    ]
    for pat in pats:
        hits = sorted(func_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def steady_state_keep_mask(confounds_file, n_trs):
    """
    Return True for retained steady-state TRs.
    fMRIPrep non_steady_state_outlier* columns mark initial volumes to remove.
    """
    c = pd.read_csv(confounds_file, sep="\t")
    if len(c) != n_trs:
        raise RuntimeError(
            f"Confounds/TR mismatch: {confounds_file}: {len(c)} vs {n_trs}"
        )
    ns = [x for x in c.columns if x.startswith("non_steady_state_outlier")]
    flagged = np.zeros(n_trs, dtype=bool)
    if ns:
        flagged = c[ns].fillna(0).to_numpy().max(axis=1) > 0
    return ~flagged


# ============================================================
# DIRECT MODAL ENERGY LOADER
#
# This replaces the expensive recursive rglob() search.
# Same direct-loading logic as previous successful null.
# ============================================================

def load_energy(
    sub_dir,
    run,
    hemi
):

    candidates = [
        sub_dir
        / f"run-{run}"
        / f"A_{hemi}.npy",

        sub_dir
        / f"run-{run:02d}"
        / f"A_{hemi}.npy",
    ]

    for f in candidates:

        if not f.exists():
            continue

        A = np.asarray(
            np.load(f),
            float
        )

        if A.ndim != 2:
            return None

        return A ** 2

    return None


# ============================================================
# LOAD NATIVE TOKEN TABLES
# ============================================================

trans = pd.read_csv(
    TRANS_CSV
)

surp = pd.read_csv(
    SURP_CSV
)


if "same_run" in trans.columns:

    trans = trans.loc[
        trans["same_run"] == 1
    ].copy()


trans["run_id"] = (
    trans["run_id"]
    .astype(int)
)

surp["run_id"] = (
    surp["run_id"]
    .astype(int)
)


# ============================================================
# PREDICTOR DEFINITIONS
# ============================================================

PREDICTORS = [
    "Qwen surprisal",
    "Curvature",
    "Token shift",
    "AR trajectory residual",
    "Subspace exit",
]


PRED_INFO = {

    "Qwen surprisal": {
        "table": surp,
        "column": "surprisal_sum",
    },

    "Curvature": {
        "table": trans,
        "column": "curvature",
    },

    "Token shift": {
        "table": trans,
        "column": "shift",
    },

    "AR trajectory residual": {
        "table": trans,
        "column": "pred_error_ar",
    },

    "Subspace exit": {
        "table": trans,
        "column": "pred_error_subspace",
    },
}


# ============================================================
# PREPARE NATIVE EVENT SEQUENCES
#
# Each predictor keeps its OWN finite event positions.
# We do NOT restrict all predictors to their intersection.
# ============================================================

events = {
    pred: {}
    for pred in PREDICTORS
}


print(
    "=" * 90
)

print(
    "NATIVE TOKEN EVENT SETS"
)

print(
    "=" * 90
)


for pred in PREDICTORS:

    table = PRED_INFO[
        pred
    ][
        "table"
    ]

    col = PRED_INFO[
        pred
    ][
        "column"
    ]

    total = 0

    for run_id in range(
        1,
        10
    ):

        d = (
            table.loc[
                table["run_id"]
                == run_id
            ]
            .sort_values("onset")
            .copy()
        )

        onsets = d[
            "onset"
        ].to_numpy(float)

        vals = d[
            col
        ].to_numpy(float)

        good = (
            np.isfinite(onsets)
            &
            np.isfinite(vals)
        )

        onsets = onsets[
            good
        ]

        vals = vals[
            good
        ]

        events[pred][run_id] = {
            "onsets": onsets,
            "values_raw": vals,
        }

        total += len(vals)

    print(
        f"{pred:25s} "
        f"N={total}"
    )


# ============================================================
# BUILD VALID SUBJECT/RUN RECORDS
# ============================================================

subjects = sorted([
    p.name
    for p in FUNC_DIR.glob(
        "sub-*"
    )
    if p.is_dir()
])


records = []


for subject in subjects:

    func_dir = (
        FUNC_DIR
        / subject
        / "func"
    )

    if not func_dir.exists():
        continue

    mapping = (
        get_subject_run_mapping(
            func_dir
        )
    )

    if not mapping:
        continue

    for ann_run, actual_run in mapping.items():

        # LPP stimulus contains annotation runs 1-9.
        # Ignore unmatched extra actual runs.
        if ann_run > 9:
            continue

        lf = find_left_func_file(
            func_dir,
            actual_run
        )

        if lf is None:
            continue

        n_trs = get_n_trs(
            lf
        )

        # Excludes pathological 7-TR run.
        if n_trs < MIN_TRS:
            continue

        # Corrected modal amplitudes are the authoritative neural time grid.
        # They were generated upstream after removal of fMRIPrep non-steady-state
        # volumes and nuisance regression. Some original confounds TSVs are not
        # present in this analysis tree (notably EN057 runs 16-23), so the null
        # generator must not re-require those source files.
        aL = load_energy(MODAL_ROOT / subject, actual_run, "L")
        aR = load_energy(MODAL_ROOT / subject, actual_run, "R")
        if aL is None or aR is None:
            continue

        # load_energy returns modal energy with the stored orientation unchanged.
        # Corrected files are K x T; accept T x K defensively.
        def _ntime(E):
            if E.shape[0] == 120:
                return E.shape[1]
            if E.shape[1] == 120:
                return E.shape[0]
            raise RuntimeError(
                f"Cannot identify modal/time axes: {subject} run {actual_run}, shape={E.shape}"
            )

        nL = _ntime(aL)
        nR = _ntime(aR)
        if nL != nR:
            raise RuntimeError(
                f"L/R corrected TR mismatch: {subject} run {actual_run}: {nL} vs {nR}"
            )

        n_final = int(nL)
        n_removed = int(n_trs - n_final)

        # The corrected amplitude pipeline removed only the initial fMRIPrep
        # non-steady-state volumes before nuisance regression. Therefore the
        # matching linguistic grid is obtained by constructing/HRF-convolving
        # on the full scanner grid and then dropping the same leading volumes.
        if n_removed < 0 or n_removed > 10:
            raise RuntimeError(
                f"Unexpected corrected length: {subject} run {actual_run}: "
                f"original={n_trs}, corrected={n_final}, removed={n_removed}"
            )

        keep_mask = np.ones(n_trs, dtype=bool)
        if n_removed:
            keep_mask[:n_removed] = False

        records.append({
            "subject": subject,
            "ann_run": ann_run,
            "actual_run": actual_run,
            "n_trs": n_trs,
            "n_trs_final": n_final,
            "n_removed_initial": n_removed,
            "keep_mask": keep_mask,
        })


records = pd.DataFrame(
    records
)


print(
    "\nValid subject-runs:",
    len(records)
)

print(
    "Participants:",
    records[
        "subject"
    ].nunique()
)


# ============================================================
# LOAD MODAL ENERGY + FIXED WORD-RATE NUISANCE
# ============================================================

datasets = []


print(
    "\nLoading modal energy and fixed word-rate nuisance..."
)


for ii, row in records.iterrows():

    subject = row[
        "subject"
    ]

    ann_run = int(
        row["ann_run"]
    )

    actual_run = int(
        row["actual_run"]
    )

    n_trs = int(
        row["n_trs"]
    )


    # --------------------------------------------------------
    # Existing empirical word-rate regressor
    # --------------------------------------------------------

    word_file = (
        REG_ROOT
        / "word_rate_hrf_per_subject"
        / f"{subject}_run-{actual_run}.npy"
    )

    if not word_file.exists():

        word_file = (
            REG_ROOT
            / "word_rate_hrf_per_subject"
            / f"{subject}_run-{actual_run:02d}.npy"
        )

    if not word_file.exists():

        raise RuntimeError(
            f"Missing word-rate regressor: "
            f"{subject} run {actual_run}"
        )


    word = np.asarray(
        np.load(word_file),
        float
    ).squeeze()


    if len(word) != n_trs:

        raise RuntimeError(
            f"Word-rate length mismatch: "
            f"{subject} run {actual_run}: "
            f"{len(word)} vs {n_trs}"
        )


    # --------------------------------------------------------
    # Apply the exact non-steady-state trimming used upstream.
    # HRF convolution remains defined on the original scanner grid.
    # --------------------------------------------------------
    keep_mask = row["keep_mask"]
    n_trs_final = int(row["n_trs_final"])
    word = word[keep_mask]

    # --------------------------------------------------------
    # Fixed nuisance design:
    # intercept + correctly aligned word rate
    # --------------------------------------------------------

    X0 = np.column_stack([
        np.ones(n_trs_final),
        word,
    ])


    Q, _ = np.linalg.qr(
        X0,
        mode="reduced"
    )


    # --------------------------------------------------------
    # Modal energy
    # --------------------------------------------------------

    hemi_data = {}


    for hemi in [
        "L",
        "R"
    ]:

        E = load_energy(
            MODAL_ROOT
            / subject,
            actual_run,
            hemi
        )


        if E is None:

            raise RuntimeError(
                f"Could not load modal energy: "
                f"{subject} run {actual_run} "
                f"hemi {hemi}"
            )


        if E.ndim != 2:

            raise RuntimeError(
                f"Unexpected E shape: "
                f"{subject} run {actual_run} "
                f"{hemi}: {E.shape}"
            )


        # Orient K x T
        if E.shape[1] == n_trs_final:

            pass

        elif E.shape[0] == n_trs_final:

            E = E.T

        else:

            raise RuntimeError(
                f"E/TR mismatch: "
                f"{subject} run {actual_run} "
                f"{hemi}: "
                f"E={E.shape}, "
                f"T={n_trs}"
            )


        # K x T -> T x K
        Y = E.T


        # ----------------------------------------------------
        # Residualize modal energy against fixed nuisance
        # ----------------------------------------------------

        Y_res = (
            Y
            -
            Q @ (
                Q.T @ Y
            )
        )


        # Needed for standardized partial beta*
        sd_E = np.std(
            Y,
            axis=0,
            ddof=1
        )


        hemi_data[
            hemi
        ] = {
            "Y_res": Y_res,
            "sd_E": sd_E,
        }


    datasets.append({
        "subject": subject,
        "ann_run": ann_run,
        "actual_run": actual_run,
        "n_trs": n_trs,
        "n_trs_final": n_trs_final,
        "keep_mask": keep_mask,
        "Q": Q,
        "hemi": hemi_data,
    })


    if (
        len(datasets) % 50 == 0
        or len(datasets) == len(records)
    ):

        print(
            f"  loaded "
            f"{len(datasets)}/{len(records)} "
            f"subject-runs"
        )


print(
    "\nLoaded datasets:",
    len(datasets)
)


# ============================================================
# SUBJECT / MODE INDEX
# ============================================================

subjects_valid = sorted(
    set(
        d["subject"]
        for d in datasets
    )
)


subject_index = {
    s: i
    for i, s in enumerate(
        subjects_valid
    )
}


N_SUB = len(
    subjects_valid
)


K = len(
    datasets[0][
        "hemi"
    ][
        "L"
    ][
        "sd_E"
    ]
)


# Drop global mode 0 exactly as in standardized profiles.
N_MODE = K - 1


print(
    "Participants loaded:",
    N_SUB
)

print(
    "K:",
    K
)

print(
    "Non-global modes:",
    N_MODE
)


# ============================================================
# TOKEN REGRESSOR BUILDER
#
# lag = 0:
#     exact empirical construction
#
# lag != 0:
#     rotate native finite token values FIRST,
#     while retaining original eligible onsets,
#     THEN perform empirical HRF pipeline.
# ============================================================

def build_token_regressor(
    pred,
    ann_run,
    n_trs,
    lag=0
):

    ev = events[
        pred
    ][
        ann_run
    ]


    onsets = ev[
        "onsets"
    ]


    vals = ev[
        "values_raw"
    ].copy()


    if lag != 0:

        vals = np.roll(
            vals,
            int(lag)
        )


    # --------------------------------------------------------
    # EXACT empirical order:
    #
    # 1. center native values within annotation run
    # 2. weighted impulse train
    # 3. HRF convolution
    # 4. z-score convolved regressor
    # --------------------------------------------------------

    vals = (
        vals
        - np.mean(vals)
    )


    x = build_impulse_regressor(
        onsets,
        vals,
        n_trs=n_trs,
        tr=TR,
    )


    x = convolve_hrf(
        x,
        HRF
    )


    x = zscore_safe(
        x
    )


    return x


# ============================================================
# SAVED EMPIRICAL REGRESSOR DIRECTORIES
# ============================================================

SAVED_REG_DIR = {

    "Qwen surprisal":
        REG_ROOT
        / "surprisal_sum_centered_hrf_per_subject",

    "Curvature":
        REG_ROOT
        / "curvature_centered_hrf_per_subject",

    "Token shift":
        REG_ROOT
        / "shift_centered_hrf_per_subject",

    "AR trajectory residual":
        REG_ROOT
        / "pred_error_ar_centered_hrf_per_subject",

    "Subspace exit":
        REG_ROOT
        / "pred_error_subspace_centered_hrf_per_subject",
}


# ============================================================
# EMPIRICAL REGRESSOR RECONSTRUCTION QC
#
# Rebuild every empirical predictor from the native token table
# and compare against the saved HRF regressor.
#
# This MUST pass before trusting the null.
# ============================================================

emp_reg_cache = {}


print(
    "\n"
    + "=" * 90
)

print(
    "UPSTREAM EMPIRICAL REGRESSOR RECONSTRUCTION QC"
)

print(
    "=" * 90
)


qc_rows = []


for d in datasets:

    subject = d[
        "subject"
    ]

    ann_run = d[
        "ann_run"
    ]

    actual_run = d[
        "actual_run"
    ]

    n_trs = d[
        "n_trs"
    ]


    for pred in PREDICTORS:

        x = build_token_regressor(
            pred,
            ann_run,
            n_trs,
            lag=0
        )


        key = (
            pred,
            subject,
            actual_run
        )


        x_full = x.copy()

        saved_file = (
            SAVED_REG_DIR[pred]
            / f"{subject}_run-{actual_run}.npy"
        )


        if not saved_file.exists():

            saved_file = (
                SAVED_REG_DIR[pred]
                / f"{subject}_run-{actual_run:02d}.npy"
            )


        if not saved_file.exists():

            raise RuntimeError(
                f"Missing saved empirical regressor: "
                f"{pred}, "
                f"{subject}, "
                f"run {actual_run}"
            )


        saved = np.asarray(
            np.load(
                saved_file
            ),
            float
        ).squeeze()


        if len(saved) != len(x_full):

            raise RuntimeError(
                f"Saved regressor length mismatch: "
                f"{pred}, "
                f"{subject}, "
                f"run {actual_run}"
            )


        if (
            np.std(x_full) == 0
            or np.std(saved) == 0
        ):

            r = np.nan

        else:

            r = np.corrcoef(
                x_full,
                saved
            )[0, 1]


        # Cache the empirical predictor on the corrected steady-state grid.
        emp_reg_cache[key] = x_full[d["keep_mask"]]

        qc_rows.append({
            "predictor": pred,
            "r": r,
            "max_abs_diff":
                float(
                    np.max(
                        np.abs(
                            x - saved
                        )
                    )
                ),
        })


qc = pd.DataFrame(
    qc_rows
)


qc_summary = (
    qc
    .groupby(
        "predictor"
    )
    .agg(
        min_r=(
            "r",
            "min"
        ),
        median_r=(
            "r",
            "median"
        ),
        max_abs_diff=(
            "max_abs_diff",
            "max"
        ),
    )
)


print(
    qc_summary.to_string()
)


# ============================================================
# PROFILE ESTIMATION
#
# Model:
#
#   E_k ~ 1 + word_rate + predictor
#
# FWL:
#   residualize E_k and predictor against
#   [intercept, word rate]
#
# Standardized partial coefficient:
#
#   beta*_k = beta_k * SD(X) / SD(E_k)
#
# Aggregation:
#   run -> participant -> hemisphere average -> group
# ============================================================

def estimate_profile(
    pred,
    lag_by_run=None
):

    # participant x hemisphere x non-global mode
    sums = np.zeros(
        (
            N_SUB,
            2,
            N_MODE
        ),
        dtype=float
    )


    counts = np.zeros(
        (
            N_SUB,
            2
        ),
        dtype=int
    )


    # Shifted linguistic regressors depend only on:
    # annotation run, nTR, and lag.
    #
    # Participants hearing the same narrative run with the same
    # nTR therefore reuse the same shifted HRF regressor.
    x_cache = {}


    for d in datasets:

        subject = d[
            "subject"
        ]

        si = subject_index[
            subject
        ]

        ann_run = d[
            "ann_run"
        ]

        actual_run = d[
            "actual_run"
        ]

        n_trs = d[
            "n_trs"
        ]

        Q = d[
            "Q"
        ]


        # ----------------------------------------------------
        # Empirical or shifted linguistic regressor
        # ----------------------------------------------------

        if lag_by_run is None:

            x = emp_reg_cache[
                (
                    pred,
                    subject,
                    actual_run
                )
            ]


        else:

            lag = int(
                lag_by_run[
                    ann_run
                ]
            )


            cache_key = (
                ann_run,
                n_trs,
                lag
            )


            if cache_key not in x_cache:

                x_cache[
                    cache_key
                ] = build_token_regressor(
                    pred,
                    ann_run,
                    n_trs,
                    lag=lag
                )


            x = x_cache[
                cache_key
            ]

            # The surrogate is generated and HRF-convolved on the original
            # scanner grid, then the same non-steady-state TRs are removed.
            x = x[d["keep_mask"]]


        # ----------------------------------------------------
        # FWL residualization of linguistic predictor
        # ----------------------------------------------------

        x_res = (
            x
            -
            Q @ (
                Q.T @ x
            )
        )


        denom = float(
            x_res @ x_res
        )


        if (
            not np.isfinite(denom)
            or denom <= 0
        ):
            continue


        # Standardization must use the predictor on the same corrected
        # steady-state TR grid used by the GLM.
        sd_x = np.std(
            x,
            ddof=1
        )


        if (
            not np.isfinite(sd_x)
            or sd_x == 0
        ):
            continue


        # ----------------------------------------------------
        # Hemispheres
        # ----------------------------------------------------

        for hi, hemi in enumerate([
            "L",
            "R"
        ]):

            Y_res = d[
                "hemi"
            ][
                hemi
            ][
                "Y_res"
            ]


            sd_E = d[
                "hemi"
            ][
                hemi
            ][
                "sd_E"
            ]


            # beta for all K modes simultaneously
            beta = (
                x_res
                @ Y_res
            ) / denom


            beta_star = (
                beta
                * sd_x
                / sd_E
            )


            # Drop mode 0
            sums[
                si,
                hi,
                :
            ] += beta_star[
                1:
            ]


            counts[
                si,
                hi
            ] += 1


    # --------------------------------------------------------
    # Run -> participant
    # --------------------------------------------------------

    if np.any(
        counts == 0
    ):

        bad = np.argwhere(
            counts == 0
        )

        raise RuntimeError(
            f"Zero run counts encountered: {bad}"
        )


    subj_profiles = (
        sums
        /
        counts[
            :,
            :,
            None
        ]
    )


    # --------------------------------------------------------
    # Participant -> group, separately by hemisphere
    # --------------------------------------------------------

    group_hemi = np.mean(
        subj_profiles,
        axis=0
    )


    # --------------------------------------------------------
    # Hemisphere average
    # --------------------------------------------------------

    group_profile = np.mean(
        group_hemi,
        axis=0
    )


    return group_profile


# ============================================================
# EMPIRICAL PROFILE RECONSTRUCTION
# ============================================================

print(
    "\nReconstructing empirical beta* profiles..."
)


emp_profiles = {}


for pred in PREDICTORS:

    emp_profiles[
        pred
    ] = estimate_profile(
        pred
    )

    print(
        f"  {pred}"
    )


# ============================================================
# HISTORICAL SAVED-PROFILE QC OMITTED IN CORRECTED ANALYSIS
# ============================================================
# The saved standardized profiles belong to the historical uncorrected
# modal-energy analysis. The corrected empirical profiles above are therefore
# the authoritative empirical target for this nuisance-corrected null.
# ============================================================

# ============================================================
# EMPIRICAL TEST STATISTICS
#
# Primary:
#   profile RMS
#
# Secondary:
#   mean absolute beta*
#
# We do NOT use the five-predictor bundle statistic here.
# ============================================================

def mean_abs(x):

    return float(
        np.mean(
            np.abs(x)
        )
    )


def rms(x):

    return float(
        np.sqrt(
            np.mean(
                np.asarray(x) ** 2
            )
        )
    )


emp_stats = {}


print(
    "\n"
    + "=" * 90
)

print(
    "EMPIRICAL STATISTICS"
)

print(
    "=" * 90
)


for pred in PREDICTORS:

    p = emp_profiles[
        pred
    ]


    emp_stats[
        pred
    ] = {

        "mean_beta_star":
            float(
                np.mean(p)
            ),

        "mean_abs_beta_star":
            mean_abs(p),

        "profile_rms":
            rms(p),
    }


    print(
        f"{pred:25s} "
        f"mean={np.mean(p): .6f}   "
        f"mean|beta*|={mean_abs(p):.6f}   "
        f"RMS={rms(p):.6f}"
    )


# ============================================================
# SIMPLE INDEPENDENT UPSTREAM-OF-HRF NULLS
# CHECKPOINTED + RESUMABLE
# ============================================================

SAVE_EVERY = 100  # N_PERM is defined once at the top (or via environment variable)

# IMPORTANT:
# fixed seed + deterministic regeneration of RNG state
# makes resumed run identical to uninterrupted run
SEED = 20260915


# ============================================================
# OUTPUT FILES
# ============================================================

NULL_FILE = (
    OUT_ROOT
    / f"token_upstream_hrf_null_{N_PERM}.csv"
)

SUMMARY_FILE = (
    OUT_ROOT
    / f"token_upstream_hrf_null_{N_PERM}_summary.csv"
)


# ============================================================
# LOAD EXISTING CHECKPOINT IF PRESENT
# ============================================================

if NULL_FILE.exists():

    existing = pd.read_csv(
        NULL_FILE
    )

    if len(existing) > N_PERM:

        raise RuntimeError(
            f"Existing file contains "
            f"{len(existing)} permutations, "
            f"but N_PERM={N_PERM}"
        )

    completed = len(
        existing
    )

    results = (
        existing
        .to_dict(
            orient="records"
        )
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "RESUMING EXISTING NULL"
    )

    print(
        "=" * 90
    )

    print(
        f"Found {completed} completed "
        f"permutations in:"
    )

    print(
        NULL_FILE
    )

else:

    completed = 0
    results = []

    print(
        "\nNo existing checkpoint found."
    )

    print(
        "Starting new null."
    )


# ============================================================
# RECREATE RNG STATE DETERMINISTICALLY
#
# We deliberately regenerate the random draws corresponding
# to already completed permutations.
#
# This guarantees that:
#
# uninterrupted 10,000-run
#
# and
#
# interrupted + resumed 10,000-run
#
# generate exactly the same permutation sequence.
# ============================================================

rng = np.random.default_rng(
    SEED
)


def draw_lags_for_predictor(
    pred,
    rng
):

    lag_by_run = {}

    for run_id in range(
        1,
        10
    ):

        n = len(
            events[
                pred
            ][
                run_id
            ][
                "values_raw"
            ]
        )


        min_shift = max(
            1,
            int(
                np.floor(
                    MIN_SHIFT_FRAC
                    * n
                )
            )
        )


        low = min_shift
        high = n - min_shift


        if high < low:

            raise RuntimeError(
                f"No legal shifts for "
                f"{pred}, run {run_id}, "
                f"N={n}"
            )


        lag_by_run[
            run_id
        ] = int(
            rng.integers(
                low,
                high + 1
            )
        )


    return lag_by_run


# ------------------------------------------------------------
# Advance RNG through permutations already completed.
#
# No GLMs are recomputed here; we only reproduce/discard
# their random lag draws, which is essentially instantaneous.
# ------------------------------------------------------------

if completed > 0:

    print(
        f"\nAdvancing RNG through "
        f"{completed} completed permutations..."
    )

    for _ in range(
        completed
    ):

        for pred in PREDICTORS:

            _ = draw_lags_for_predictor(
                pred,
                rng
            )

    print(
        "RNG state restored."
    )


# ============================================================
# RUN REMAINING PERMUTATIONS
# ============================================================

start_all = time.time()


print(
    "\n"
    + "=" * 90
)

print(
    f"TARGET: {N_PERM} UPSTREAM-OF-HRF PERMUTATIONS"
)

print(
    f"ALREADY COMPLETE: {completed}"
)

print(
    f"REMAINING: {N_PERM - completed}"
)

print(
    f"CHECKPOINT EVERY: {SAVE_EVERY}"
)

print(
    "=" * 90
)


for perm_idx in range(
    completed,
    N_PERM
):

    row = {
        "perm": perm_idx + 1
    }


    t0 = time.time()


    # ========================================================
    # Each predictor receives its own independent circular
    # displacement within each narrative run.
    # ========================================================

    for pred in PREDICTORS:

        lag_by_run = (
            draw_lags_for_predictor(
                pred,
                rng
            )
        )


        null_profile = (
            estimate_profile(
                pred,
                lag_by_run=
                    lag_by_run
            )
        )


        safe = (
            pred
            .lower()
            .replace(
                " ",
                "_"
            )
            .replace(
                "-",
                "_"
            )
        )


        row[
            f"{safe}_mean_beta_star"
        ] = float(
            np.mean(
                null_profile
            )
        )


        row[
            f"{safe}_mean_abs_beta_star"
        ] = mean_abs(
            null_profile
        )


        row[
            f"{safe}_profile_rms"
        ] = rms(
            null_profile
        )


    row[
        "seconds"
    ] = (
        time.time()
        - t0
    )


    results.append(
        row
    )


    current_n = (
        perm_idx + 1
    )


    # ========================================================
    # CHECKPOINT
    # ========================================================

    if (
        current_n % SAVE_EVERY == 0
        or current_n == N_PERM
    ):

        checkpoint_df = (
            pd.DataFrame(
                results
            )
        )


        # Write temporary file first, then atomically replace.
        # Prevents corruption if the process dies during write.
        tmp_file = (
            NULL_FILE
            .with_suffix(
                ".tmp.csv"
            )
        )


        checkpoint_df.to_csv(
            tmp_file,
            index=False
        )


        tmp_file.replace(
            NULL_FILE
        )


        elapsed = (
            time.time()
            - start_all
        )


        newly_done = (
            current_n
            - completed
        )


        sec_per_perm = (
            elapsed
            / newly_done
        )


        remaining = (
            N_PERM
            - current_n
        )


        eta_sec = (
            sec_per_perm
            * remaining
        )


        print(
            f"Checkpoint: "
            f"{current_n:5d}/{N_PERM} | "
            f"{sec_per_perm:.3f} s/perm | "
            f"ETA {eta_sec / 60:.1f} min"
        )


# ============================================================
# FINAL NULL DATAFRAME
# ============================================================

null_df = pd.DataFrame(
    results
)


# ============================================================
# PERMUTATION P VALUE
# ============================================================

def p_greater(
    empirical,
    values
):

    values = np.asarray(
        values,
        float
    )


    return float(
        (
            1
            +
            np.sum(
                values >= empirical
            )
        )
        /
        (
            len(values)
            + 1
        )
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = []


for pred in PREDICTORS:

    safe = (
        pred
        .lower()
        .replace(
            " ",
            "_"
        )
        .replace(
            "-",
            "_"
        )
    )


    for stat in [
        "profile_rms",
        "mean_abs_beta_star",
    ]:

        col = (
            f"{safe}_{stat}"
        )


        vals = null_df[
            col
        ].to_numpy(float)


        empirical = (
            emp_stats[
                pred
            ][
                stat
            ]
        )


        summary.append({

            "predictor":
                pred,

            "metric":
                stat,

            "empirical":
                empirical,

            "null_mean":
                float(
                    np.mean(vals)
                ),

            "null_sd":
                float(
                    np.std(
                        vals,
                        ddof=1
                    )
                ),

            "null_2p5":
                float(
                    np.percentile(
                        vals,
                        2.5
                    )
                ),

            "null_97p5":
                float(
                    np.percentile(
                        vals,
                        97.5
                    )
                ),

            "p_perm_greater":
                p_greater(
                    empirical,
                    vals
                ),
        })


summary_df = pd.DataFrame(
    summary
)


# ============================================================
# SAVE FINAL SUMMARY
# ============================================================

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


print(
    "\n"
    + "=" * 110
)

print(
    "UPSTREAM-OF-HRF TOKEN NULL COMPLETE"
)

print(
    "=" * 110
)


print(
    summary_df
    .round(6)
    .to_string(
        index=False
    )
)


print(
    "\nFinal permutation file:"
)

print(
    NULL_FILE
)


print(
    "\nFinal summary file:"
)

print(
    SUMMARY_FILE
)
