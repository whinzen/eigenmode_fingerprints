#!/usr/bin/env python3
"""
Corrected sentence-shift shuffle null.

Purpose
-------
Re-run the historical sentence-shift stimulus-specific validation using the
nuisance-corrected modal amplitudes, while reproducing the authoritative
corrected 8-predictor sentence GLM exactly.

Authoritative corrected empirical model
---------------------------------------
Outcome:
    E_k(t) = A_k(t)^2

Predictors:
    1. sentence onset
    2. word rate
    3. content-word density
    4. preceding pause
    5. sentence shift
    6. acoustic envelope
    7. mean log-F0
    8. pitch reset

The saved full-length regressors are loaded from
pang_out/regressors_sentence_controls_exact and the exact fMRIPrep
non-steady-state mask is applied AFTER loading them, matching the code that
generated corrected_sentence_effects_run_level_FIXED.csv.

Sentence-shift null
-------------------
Historical null logic is retained:
    - sentence-shift values are mean-centered within stimulus section
    - values are randomly reassigned among the TRUE sentence-transition
      locations within that section
    - event locations therefore remain fixed
    - the same shuffled mapping is used for all participants hearing a section
    - the shuffled impulse series is HRF-convolved on the ORIGINAL full grid
    - only then is the exact non-steady-state mask applied
    - the other seven predictors remain fixed

Three mandatory QC stages precede permutation inference:
    QC1: direct 8-predictor GLM reproduces authoritative corrected raw shift
         betas at machine precision
    QC2: Frisch-Waugh-Lovell implementation reproduces the direct GLM
    QC3: empirical beta* from the FWL implementation is finite and has the
         expected 439-run / 49-participant structure

Default N_PERM=10 for QC.
Only change to 10000 after QC passes.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import gamma


# ============================================================
# SETTINGS
# ============================================================

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

N_PERM = int(os.environ.get("N_PERM", "10"))
SEED = 20260914
TR = 2.0
MIN_TRS = 100
K_EXPECTED = 120

# Historical event table
# Qwen3-4B events used for empirical + null analysis
EVENT_FILE = (
    PANG
    / "sentence_level_controls_qwen3_4b_proportional"
    / "sentence_shift_qwen3_4b_continuous.csv"
)

# Historical Qwen3-0.6B events used ONLY to recover and verify
# participant acquisition-run -> stimulus-section mapping
MAPPING_EVENT_FILE = (
    PANG
    / "sentence_level_controls"
    / "sentence_shift_qwen_continuous.csv"
)

# Historical/full-length sentence regressors
REG_ROOT = (
    PANG
    / "regressors_sentence_controls_exact"
)

SENT_DIRS = {
    "sentence_onset":
        REG_ROOT / "sentence_onset_hrf",

    "word_rate":
        REG_ROOT / "word_rate_hrf",

    "content_density":
        REG_ROOT / "content_density_hrf",

    "preceding_pause":
        REG_ROOT / "preceding_pause_centered_hrf",

    "sentence_shift":
        REG_ROOT / "sentence_shift_hrf",

    "acoustic_envelope":
        REG_ROOT / "acoustic_envelope_hrf",

    "mean_log_f0":
        REG_ROOT / "mean_log_f0_hrf",

    "pitch_reset":
        REG_ROOT / "sentence_pitch_reset_hrf",
}

SENT_PREDICTORS = [
    "sentence_onset",
    "word_rate",
    "content_density",
    "preceding_pause",
    "sentence_shift",
    "acoustic_envelope",
    "mean_log_f0",
    "pitch_reset",
]

SHIFT_INDEX = SENT_PREDICTORS.index("sentence_shift")

# Corrected neural amplitudes
CORR_ROOT = (
    PANG
    / "nuisance_robustness"
)

A_ROOT = (
    CORR_ROOT
    / "corrected_modal_amplitudes"
)

# Authoritative corrected raw sentence betas
AUTH_FILE = (
    CORR_ROOT
    / "corrected_validated_linguistic_effects"
    / "corrected_sentence_effects_run_level_FIXED.csv"
)

# Original empirical fMRIPrep tree, needed only for exact nonsteady masks
EMPIRICAL = (
    BASE
    / "data"
    / "empirical"
)

OUT_DIR = (
    CORR_ROOT
    / "figure3_corrected"
    / "sentence_shift_shuffle_null_qwen3_4b_proportional"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# HISTORICAL HRF
# ============================================================

def canonical_hrf(tr=2.0, duration=32.0):
    """
    Historical HRF used by the sentence-shift null generator:
        gamma(t, 6) - 0.5 * gamma(t, 16)
    normalized by maximum absolute value.
    """
    t = np.arange(
        0.0,
        duration,
        tr,
        dtype=float,
    )

    h = (
        gamma.pdf(t, 6)
        - 0.5 * gamma.pdf(t, 16)
    )

    m = np.max(
        np.abs(h)
    )

    if m > 0:
        h = h / m

    return h


HRF = canonical_hrf(
    TR
)


def zscore(x):
    x = np.asarray(
        x,
        float,
    )

    sd = np.std(
        x,
        ddof=1,
    )

    if (
        not np.isfinite(sd)
        or sd == 0
    ):
        return np.zeros_like(
            x,
            dtype=float,
        )

    return (
        x - np.mean(x)
    ) / sd


# ============================================================
# FILE LOOKUP HELPERS
# ============================================================

def find_regressor(
    folder,
    subject,
    run,
):

    candidates = [
        folder
        / f"{subject}_run-{run}.npy",

        folder
        / f"{subject}_run-{run:02d}.npy",
    ]

    for f in candidates:
        if f.exists():
            return f

    return None


def find_confounds(
    subject,
    run,
):

    hits = sorted(
        EMPIRICAL.glob(
            f"{subject}/**/"
            f"{subject}_task-lppEN_run-{run}_"
            f"desc-confounds_timeseries.tsv"
        )
    )

    if len(hits) == 0:

        hits = sorted(
            EMPIRICAL.glob(
                f"{subject}/**/"
                f"{subject}_task-lppEN_run-{run:02d}_"
                f"desc-confounds_timeseries.tsv"
            )
        )

    if len(hits) == 0:
        raise FileNotFoundError(
            f"No confounds for {subject} run-{run}"
        )

    return hits[0]


def get_nonsteady_keep_mask(
    confounds_file,
):

    conf = pd.read_csv(
        confounds_file,
        sep="\t",
    )

    ns_cols = [
        c
        for c in conf.columns
        if c.startswith(
            "non_steady_state_outlier"
        )
    ]

    keep = np.ones(
        len(conf),
        dtype=bool,
    )

    if len(ns_cols) > 0:

        ns = (
            conf[ns_cols]
            .fillna(0)
            .to_numpy(float)
        )

        keep = ~np.any(
            ns != 0,
            axis=1,
        )

    return keep


# ============================================================
# BUILD AUTHORITATIVE 439-RUN GRID DIRECTLY
# ============================================================

auth = pd.read_csv(
    AUTH_FILE
)

required_auth = {
    "subject",
    "run",
    "hemisphere",
    "mode",
    "predictor",
    "beta",
}

missing_auth = (
    required_auth
    - set(auth.columns)
)

if missing_auth:
    raise RuntimeError(
        f"Authoritative table missing columns: "
        f"{sorted(missing_auth)}"
    )

auth_shift = (
    auth[
        auth["predictor"]
        == "Sentence shift"
    ]
    .copy()
)

auth_grid = (
    auth_shift[
        [
            "subject",
            "run",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "subject",
            "run",
        ]
    )
    .reset_index(drop=True)
)

print(
    "Authoritative corrected grid:",
    f"{len(auth_grid)} runs,",
    f"{auth_grid['subject'].nunique()} participants,",
    f"{auth_shift[['subject','run','hemisphere']].drop_duplicates().shape[0]} hemisphere-runs",
)

assert len(auth_grid) == 439
assert auth_grid["subject"].nunique() == 49


# ============================================================
# HISTORICAL SENTENCE-SHIFT EVENTS
# ============================================================

events = pd.read_csv(
    EVENT_FILE
)

if "stimulus_run" in events.columns:
    event_run_col = "stimulus_run"
elif "run_id" in events.columns:
    event_run_col = "run_id"
else:
    raise RuntimeError(
        "Cannot identify stimulus-run column "
        f"in {EVENT_FILE}"
    )

if "sentence_onset" in events.columns:
    onset_col = "sentence_onset"
elif "onset" in events.columns:
    onset_col = "onset"
else:
    raise RuntimeError(
        "Cannot identify sentence-onset column"
    )

shift_col = "sentence_shift"

if shift_col not in events.columns:
    raise RuntimeError(
        f"{shift_col} absent from event table"
    )

events = events.copy()

events["shift_centered"] = (
    events[shift_col]
    - events.groupby(
        event_run_col
    )[shift_col].transform(
        "mean"
    )
)

event_by_run = {}

for stim_run, d in events.groupby(
    event_run_col
):

    d = d.sort_values(
        onset_col
    )

    event_by_run[
        int(stim_run)
    ] = {
        "onsets":
            d[onset_col]
            .to_numpy(float),

        "weights":
            d["shift_centered"]
            .to_numpy(float),
    }

print(
    "Stimulus runs:",
    sorted(
        event_by_run.keys()
    )
)

# ============================================================
# HISTORICAL 0.6B EVENTS — MAPPING QC ONLY
# ============================================================

mapping_events = pd.read_csv(
    MAPPING_EVENT_FILE
)

if "stimulus_run" in mapping_events.columns:
    mapping_run_col = "stimulus_run"
elif "run_id" in mapping_events.columns:
    mapping_run_col = "run_id"
else:
    raise RuntimeError(
        "Cannot identify stimulus-run column "
        f"in {MAPPING_EVENT_FILE}"
    )

if "sentence_onset" in mapping_events.columns:
    mapping_onset_col = "sentence_onset"
elif "onset" in mapping_events.columns:
    mapping_onset_col = "onset"
else:
    raise RuntimeError(
        "Cannot identify sentence-onset column "
        f"in {MAPPING_EVENT_FILE}"
    )

if "sentence_shift" not in mapping_events.columns:
    raise RuntimeError(
        "sentence_shift absent from historical mapping event table"
    )

mapping_events = mapping_events.copy()

mapping_events["shift_centered"] = (
    mapping_events["sentence_shift"]
    - mapping_events.groupby(
        mapping_run_col
    )["sentence_shift"].transform(
        "mean"
    )
)

mapping_event_by_run = {}

for stim_run, d in mapping_events.groupby(
    mapping_run_col
):

    d = d.sort_values(
        mapping_onset_col
    )

    mapping_event_by_run[
        int(stim_run)
    ] = {
        "onsets":
            d[mapping_onset_col]
            .to_numpy(float),

        "weights":
            d["shift_centered"]
            .to_numpy(float),
    }

if sorted(mapping_event_by_run.keys()) != list(range(1, 10)):
    raise RuntimeError(
        "Historical mapping events do not contain stimulus runs 1..9"
    )

print(
    "Historical mapping stimulus runs:",
    sorted(mapping_event_by_run.keys())
)


if sorted(
    event_by_run.keys()
) != list(
    range(1, 10)
):
    raise RuntimeError(
        "Expected stimulus runs 1..9"
    )


# ============================================================
# MAP PARTICIPANT ACQUISITION RUN -> STIMULUS RUN
#
# IMPORTANT:
# The final saved sentence regressors are authoritative.
#
# We identify the stimulus section for each participant-run by
# matching its SAVED empirical sentence-shift HRF regressor to
# the nine event-derived candidate regressors.
#
# This avoids assuming that acquisition-run ordering itself
# defines stimulus section, while making the mapping testable.
# ============================================================

def build_shift_regressor(
    T,
    onsets,
    weights,
):

    impulses = np.zeros(
        T,
        dtype=float,
    )

    for onset, weight in zip(
        onsets,
        weights,
    ):

        idx = int(
            np.round(
                float(onset)
                / TR
            )
        )

        if 0 <= idx < T:

            impulses[idx] += float(
                weight
            )

    reg = np.convolve(
        impulses,
        HRF,
        mode="full",
    )[:T]

    return zscore(
        reg
    )


def correlation(
    x,
    y,
):

    x = np.asarray(
        x,
        float,
    )

    y = np.asarray(
        y,
        float,
    )

    if len(x) != len(y):
        return np.nan

    if (
        np.std(x) == 0
        or np.std(y) == 0
    ):
        return np.nan

    return float(
        np.corrcoef(
            x,
            y,
        )[0, 1]
    )


# ============================================================
# PRECOMPUTE ALL RUN DATA
# ============================================================

run_data = {}

removed_counts = []

mapping_rows = []


for _, g in auth_grid.iterrows():

    subject = str(
        g["subject"]
    )

    run = int(
        g["run"]
    )

    # --------------------------------------------------------
    # Locate all eight authoritative saved regressors
    # --------------------------------------------------------

    reg_files = {}

    for pred in SENT_PREDICTORS:

        f = find_regressor(
            SENT_DIRS[pred],
            subject,
            run,
        )

        if f is None:
            raise FileNotFoundError(
                f"Missing {pred}: "
                f"{subject} run-{run}"
            )

        reg_files[pred] = f


    # --------------------------------------------------------
    # Exact nonsteady-state mask
    # --------------------------------------------------------

    conf_file = find_confounds(
        subject,
        run,
    )

    keep = get_nonsteady_keep_mask(
        conf_file
    )

    T_full = len(
        keep
    )

    n_removed = int(
        (~keep).sum()
    )

    removed_counts.append(
        n_removed
    )


    # --------------------------------------------------------
    # Load full authoritative design
    # --------------------------------------------------------

    X_full = []

    for pred in SENT_PREDICTORS:

        x = np.asarray(
            np.load(
                reg_files[pred]
            ),
            float,
        ).squeeze()

        if len(x) != T_full:
            raise RuntimeError(
                f"{subject} run-{run} {pred}: "
                f"regressor={len(x)}, "
                f"confounds={T_full}"
            )

        X_full.append(
            x
        )

    X_full = np.column_stack(
        X_full
    )

    X = X_full[
        keep,
        :
    ]


    # --------------------------------------------------------
    # Corrected modal amplitudes
    # --------------------------------------------------------

    A_by_hemi = {}

    for hemi in [
        "L",
        "R",
    ]:

        afile = (
            A_ROOT
            / subject
            / f"run-{run}"
            / f"A_{hemi}.npy"
        )

        if not afile.exists():
            raise FileNotFoundError(
                afile
            )

        A = np.asarray(
            np.load(
                afile
            ),
            float,
        )

        if A.ndim != 2:
            raise RuntimeError(
                f"{afile}: unexpected "
                f"shape {A.shape}"
            )

        if A.shape[0] != K_EXPECTED:
            raise RuntimeError(
                f"{afile}: expected "
                f"{K_EXPECTED} modes, "
                f"got {A.shape}"
            )

        if A.shape[1] != X.shape[0]:
            raise RuntimeError(
                f"{subject} run-{run} {hemi}: "
                f"A={A.shape}, X={X.shape}"
            )

        A_by_hemi[
            hemi
        ] = A


    # --------------------------------------------------------
    # Identify stimulus section by matching the SAVED
    # sentence-shift regressor against event-derived candidates.
    #
    # Compare on full grid because both historical construction
    # and saved regressor are full-length here.
    # --------------------------------------------------------

    saved_shift_full = X_full[
        :,
        SHIFT_INDEX,
    ]

    candidate_scores = {}

    for stim_run in sorted(
        mapping_event_by_run.keys()
    ):

        ev = mapping_event_by_run[
            stim_run
        ]

        candidate = build_shift_regressor(
            T_full,
            ev["onsets"],
            ev["weights"],
        )

        candidate_scores[
            stim_run
        ] = correlation(
            saved_shift_full,
            candidate,
        )

    best_stim = max(
        candidate_scores,
        key=lambda k:
            (
                -np.inf
                if not np.isfinite(
                    candidate_scores[k]
                )
                else candidate_scores[k]
            ),
    )

    best_r = candidate_scores[
        best_stim
    ]

    second_r = sorted(
        [
            v
            for v in candidate_scores.values()
            if np.isfinite(v)
        ],
        reverse=True,
    )[1]

    # --------------------------------------------------------
    # QWEN3-4B ROBUSTNESS:
    # Mapping above is anchored to the historical saved
    # sentence-shift regressor. Having identified the stimulus
    # section, reconstruct the empirical 4B sentence-shift
    # regressor and replace ONLY the sentence-shift column.
    # The other seven sentence predictors remain unchanged.
    # --------------------------------------------------------

    ev4b = event_by_run[
        best_stim
    ]

    shift_4b_full = build_shift_regressor(
        T_full,
        ev4b["onsets"],
        ev4b["weights"],
    )

    if (
        len(shift_4b_full) != T_full
        or not np.all(np.isfinite(shift_4b_full))
    ):
        raise RuntimeError(
            f"Invalid 4B sentence-shift regressor: "
            f"{subject} run-{run}"
        )

    X_full[
        :,
        SHIFT_INDEX
    ] = shift_4b_full

    X = X_full[
        keep,
        :
    ]

    mapping_rows.append({
        "subject":
            subject,

        "run":
            run,

        "stimulus_run":
            best_stim,

        "match_r":
            best_r,

        "second_best_r":
            second_r,

        "margin":
            best_r - second_r,

        "T_full":
            T_full,

        "T_corrected":
            int(
                keep.sum()
            ),

        "n_removed":
            n_removed,
    })


    run_data[
        (
            subject,
            run,
        )
    ] = {
        "subject":
            subject,

        "run":
            run,

        "stimulus_run":
            best_stim,

        "keep":
            keep,

        "T_full":
            T_full,

        "X":
            X,

        "X_full":
            X_full,

        "A":
            A_by_hemi,
    }


mapping_df = pd.DataFrame(
    mapping_rows
)

print(
    f"Loaded corrected grid: "
    f"{len(run_data)} runs, "
    f"{mapping_df['subject'].nunique()} participants, "
    f"{2 * len(run_data)} hemisphere-runs"
)

print(
    "\nRemoved nonsteady TRs:"
)

print(
    pd.Series(
        removed_counts,
        name="n_removed",
    )
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# STIMULUS-MAPPING QC
# ============================================================

print(
    "\nStimulus-section mapping QC:"
)

print(
    mapping_df[
        [
            "match_r",
            "margin",
        ]
    ]
    .describe()
    .to_string()
)

print(
    "\nAssigned stimulus-run counts:"
)

print(
    mapping_df[
        "stimulus_run"
    ]
    .value_counts()
    .sort_index()
    .to_string()
)


# A correct match should be essentially exact if the historical
# event construction is identical to the saved empirical shift
# regressor. We do not silently accept an approximate mapping.

bad_mapping = mapping_df[
    (
        ~np.isfinite(
            mapping_df["match_r"]
        )
    )
    |
    (
        mapping_df["match_r"]
        < 0.999999
    )
]

if len(
    bad_mapping
) > 0:

    print(
        "\nWorst stimulus mapping matches:"
    )

    print(
        mapping_df
        .sort_values(
            "match_r"
        )
        .head(20)
        .to_string(
            index=False
        )
    )

    raise RuntimeError(
        "FATAL QC: event-derived empirical sentence-shift "
        "regressor does not exactly reproduce the saved "
        "sentence_shift_hrf regressor. Do not run permutations."
    )


# ============================================================
# DIRECT 8-PREDICTOR GLM
# ============================================================

def direct_fit(
    A,
    Xpred,
):

    E = (
        np.asarray(
            A,
            float,
        )
        ** 2
    )

    Xpred = np.asarray(
        Xpred,
        float,
    )

    good = (
        np.all(
            np.isfinite(
                Xpred
            ),
            axis=1,
        )
        &
        np.all(
            np.isfinite(
                E
            ),
            axis=0,
        )
    )

    Y = E[
        :,
        good
    ].T

    Xp = Xpred[
        good,
        :
    ]

    X = np.column_stack(
        [
            np.ones(
                Xp.shape[0]
            ),
            Xp,
        ]
    )

    B, *_ = np.linalg.lstsq(
        X,
        Y,
        rcond=None,
    )

    return (
        B,
        good,
    )


direct_rows = []


for (
    subject,
    run
), d in run_data.items():

    for hemi in [
        "L",
        "R",
    ]:

        B, good = direct_fit(
            d["A"][hemi],
            d["X"],
        )

        beta_shift = B[
            SHIFT_INDEX + 1,
            :
        ]

        for mode, b in enumerate(
            beta_shift
        ):

            direct_rows.append({
                "subject":
                    subject,

                "run":
                    run,

                "hemisphere":
                    hemi,

                "mode":
                    mode,

                "beta_direct":
                    float(b),
            })


direct_df = pd.DataFrame(
    direct_rows
)


# ============================================================
# QC1 — HISTORICAL COMPARISON ONLY
#
# The authoritative saved raw betas were obtained with the
# historical Qwen3-0.6B sentence-shift predictor. They are
# retained here as a provenance comparison, but equality is
# NOT expected after replacing sentence shift with Qwen3-4B.
#
# The merge/cardinality and finiteness checks remain mandatory.
# Internal correctness of the 4B fit is tested below by QC2:
# direct 8-predictor GLM == FWL implementation.
# ============================================================

auth_cmp = (
    auth_shift[
        [
            "subject",
            "run",
            "hemisphere",
            "mode",
            "beta",
        ]
    ]
    .rename(
        columns={
            "beta":
                "beta_authoritative"
        }
    )
)

cmp1 = direct_df.merge(
    auth_cmp,
    on=[
        "subject",
        "run",
        "hemisphere",
        "mode",
    ],
    how="inner",
    validate="one_to_one",
)

expected_rows = (
    439
    * 2
    * K_EXPECTED
)

if len(cmp1) != expected_rows:
    raise RuntimeError(
        f"QC1 merge produced {len(cmp1)} rows; "
        f"expected {expected_rows}"
    )

x = cmp1[
    "beta_direct"
].to_numpy(float)

y = cmp1[
    "beta_authoritative"
].to_numpy(float)

r_direct = np.corrcoef(
    x,
    y,
)[0, 1]

diff_direct = np.abs(
    x - y
)

print(
    "\nQC1 — Qwen3-4B direct GLM vs historical "
    "Qwen3-0.6B authoritative raw shift betas "
    "(diagnostic only; equality not expected):"
)

print(
    f"r={r_direct:.12f}, "
    f"maxdiff={diff_direct.max():.12e}, "
    f"meandiff={diff_direct.mean():.12e}"
)

if (
    not np.isfinite(r_direct)
    or not np.all(np.isfinite(x))
    or not np.all(np.isfinite(y))
    or not np.all(np.isfinite(diff_direct))
):
    raise RuntimeError(
        "FATAL QC1: non-finite values in historical "
        "0.6B vs Qwen3-4B diagnostic comparison."
    )

print(
    "QC1 diagnostic passed: row correspondence and "
    "finiteness verified; coefficient equality is not "
    "required for the 4B robustness analysis."
)


# ============================================================
# FWL IMPLEMENTATION
#
# For sentence shift:
#   fixed design =
#       intercept + other seven predictors
#
# Residualize Y and shift against fixed design, then fit
# residualized shift.
#
# This must reproduce the direct 8-predictor coefficient.
# ============================================================

def prepare_fwl(
    A,
    Xpred,
):

    A = np.asarray(
        A,
        float,
    )

    Xpred = np.asarray(
        Xpred,
        float,
    )

    E = A ** 2

    good = (
        np.all(
            np.isfinite(
                Xpred
            ),
            axis=1,
        )
        &
        np.all(
            np.isfinite(
                E
            ),
            axis=0,
        )
    )

    Y = E[
        :,
        good
    ].T

    Xp = Xpred[
        good,
        :
    ]

    shift = Xp[
        :,
        SHIFT_INDEX
    ]

    fixed_cols = [
        j
        for j in range(
            Xp.shape[1]
        )
        if j != SHIFT_INDEX
    ]

    Z = np.column_stack(
        [
            np.ones(
                Xp.shape[0]
            ),
            Xp[
                :,
                fixed_cols
            ],
        ]
    )

    Q, _ = np.linalg.qr(
        Z,
        mode="reduced",
    )

    Y_res = (
        Y
        - Q @ (
            Q.T @ Y
        )
    )

    shift_res = (
        shift
        - Q @ (
            Q.T @ shift
        )
    )

    denom = float(
        shift_res
        @ shift_res
    )

    if (
        not np.isfinite(
            denom
        )
        or denom <= 0
    ):
        raise RuntimeError(
            "Degenerate empirical residualized "
            "sentence-shift regressor"
        )

    beta = (
        shift_res
        @ Y_res
    ) / denom

    sd_shift = np.std(
        shift,
        ddof=1,
    )

    sd_Y = np.std(
        Y,
        axis=0,
        ddof=1,
    )

    beta_star = (
        beta
        * sd_shift
        / sd_Y
    )

    return {
        "good":
            good,

        "Y":
            Y,

        "Y_res":
            Y_res,

        "Q":
            Q,

        "beta":
            beta,

        "beta_star":
            beta_star,

        "sd_Y":
            sd_Y,

        "fixed_cols":
            fixed_cols,
    }


fwl_rows = []

# Store precomputed quantities for permutations.
perm_cache = {}


for (
    subject,
    run
), d in run_data.items():

    perm_cache[
        (
            subject,
            run,
        )
    ] = {
        "stimulus_run":
            d["stimulus_run"],

        "keep":
            d["keep"],

        "T_full":
            d["T_full"],

        "hemi":
            {},
    }

    for hemi in [
        "L",
        "R",
    ]:

        f = prepare_fwl(
            d["A"][hemi],
            d["X"],
        )

        perm_cache[
            (
                subject,
                run,
            )
        ][
            "hemi"
        ][
            hemi
        ] = f

        for mode in range(
            K_EXPECTED
        ):

            fwl_rows.append({
                "subject":
                    subject,

                "run":
                    run,

                "hemisphere":
                    hemi,

                "mode":
                    mode,

                "beta_fwl":
                    float(
                        f["beta"][mode]
                    ),

                "beta_star":
                    float(
                        f["beta_star"][mode]
                    ),
            })


fwl_df = pd.DataFrame(
    fwl_rows
)


# ============================================================
# QC2:
# FWL RAW BETA MUST MATCH DIRECT GLM
# ============================================================

cmp2 = direct_df.merge(
    fwl_df[
        [
            "subject",
            "run",
            "hemisphere",
            "mode",
            "beta_fwl",
        ]
    ],
    on=[
        "subject",
        "run",
        "hemisphere",
        "mode",
    ],
    how="inner",
    validate="one_to_one",
)

x = cmp2[
    "beta_direct"
].to_numpy(float)

y = cmp2[
    "beta_fwl"
].to_numpy(float)

r_fwl = np.corrcoef(
    x,
    y,
)[0, 1]

diff_fwl = np.abs(
    x - y
)

print(
    "\nQC2 — FWL vs direct 8-predictor "
    "sentence-shift beta:"
)

print(
    f"r={r_fwl:.12f}, "
    f"maxdiff={diff_fwl.max():.12e}, "
    f"meandiff={diff_fwl.mean():.12e}"
)

if (
    r_fwl < 0.999999999
    or diff_fwl.max() > 1e-8
):
    raise RuntimeError(
        "FATAL QC2: FWL does not reproduce "
        "direct sentence-shift coefficients."
    )


# ============================================================
# HISTORICAL HIERARCHICAL AGGREGATION
#
# mode 0 excluded
# runs -> participant/hemi -> participants -> hemispheres
# ============================================================

def aggregate_group_profile(
    df,
    value_col,
):

    d = df[
        df["mode"].between(
            1,
            K_EXPECTED - 1,
        )
    ].copy()

    # Average runs within participant + hemisphere + mode
    sub_hemi = (
        d.groupby(
            [
                "subject",
                "hemisphere",
                "mode",
            ],
            as_index=False,
        )[
            value_col
        ]
        .mean()
    )

    # Equal participant weighting within hemisphere
    hemi = (
        sub_hemi.groupby(
            [
                "hemisphere",
                "mode",
            ],
            as_index=False,
        )[
            value_col
        ]
        .mean()
    )

    # Equal hemisphere weighting
    group = (
        hemi.groupby(
            "mode",
            as_index=False,
        )[
            value_col
        ]
        .mean()
        .sort_values(
            "mode"
        )
    )

    profile = group[
        value_col
    ].to_numpy(float)

    if len(profile) != (
        K_EXPECTED - 1
    ):
        raise RuntimeError(
            f"Expected {K_EXPECTED - 1} "
            f"nonconstant modes; got "
            f"{len(profile)}"
        )

    return profile


emp_profile = aggregate_group_profile(
    fwl_df,
    "beta_star",
)

emp_mean = float(
    np.mean(
        emp_profile
    )
)

emp_mean_abs = float(
    np.mean(
        np.abs(
            emp_profile
        )
    )
)

emp_rms = float(
    np.sqrt(
        np.mean(
            emp_profile ** 2
        )
    )
)


# ============================================================
# QC3:
# EMPIRICAL CORRECTED BETA* SUMMARY
# ============================================================

print(
    "\nQC3 — corrected empirical sentence-shift statistics:"
)

print(
    f"mean beta*     : {emp_mean:.12f}"
)

print(
    f"mean |beta*|   : {emp_mean_abs:.12f}"
)

print(
    f"profile RMS    : {emp_rms:.12f}"
)

print(
    f"participant-runs: "
    f"{fwl_df[['subject','run']].drop_duplicates().shape[0]}"
)

print(
    f"hemisphere-runs: "
    f"{fwl_df[['subject','run','hemisphere']].drop_duplicates().shape[0]}"
)

print(
    f"participants: "
    f"{fwl_df['subject'].nunique()}"
)

if not np.all(
    np.isfinite(
        emp_profile
    )
):
    raise RuntimeError(
        "FATAL QC3: non-finite empirical group profile"
    )

assert (
    fwl_df[
        [
            "subject",
            "run",
        ]
    ]
    .drop_duplicates()
    .shape[0]
    == 439
)

assert (
    fwl_df[
        [
            "subject",
            "run",
            "hemisphere",
        ]
    ]
    .drop_duplicates()
    .shape[0]
    == 878
)

assert (
    fwl_df[
        "subject"
    ].nunique()
    == 49
)


# ============================================================
# HISTORICAL COMPARISON
# ============================================================

HIST_MEAN = 0.0074207479
HIST_MEAN_ABS = 0.0074207479
HIST_RMS = 0.0078809396

print(
    "\nHistorical -> corrected:"
)

print(
    f"mean beta*:   "
    f"{HIST_MEAN:.10f} -> "
    f"{emp_mean:.10f}"
)

print(
    f"mean |beta*|: "
    f"{HIST_MEAN_ABS:.10f} -> "
    f"{emp_mean_abs:.10f}"
)

print(
    f"profile RMS:  "
    f"{HIST_RMS:.10f} -> "
    f"{emp_rms:.10f}"
)


# ============================================================
# PERMUTATION FIT
# ============================================================

def fit_permuted_shift(
    g_full,
    cache,
):

    keep = cache[
        "keep"
    ]

    g = np.asarray(
        g_full,
        float,
    )[
        keep
    ]

    rows = []

    for hemi in [
        "L",
        "R",
    ]:

        f = cache[
            "hemi"
        ][
            hemi
        ]

        # Same finite-time mask used for empirical FWL.
        g_use = g[
            f["good"]
        ]

        Q = f[
            "Q"
        ]

        Y_res = f[
            "Y_res"
        ]

        g_res = (
            g_use
            - Q @ (
                Q.T @ g_use
            )
        )

        denom = float(
            g_res
            @ g_res
        )

        if (
            not np.isfinite(
                denom
            )
            or denom <= 0
        ):
            raise RuntimeError(
                "Degenerate permuted residualized "
                "sentence-shift regressor"
            )

        beta = (
            g_res
            @ Y_res
        ) / denom

        sg = np.std(
            g_use,
            ddof=1,
        )

        sy = f[
            "sd_Y"
        ]

        beta_star = (
            beta
            * sg
            / sy
        )

        for mode in range(
            K_EXPECTED
        ):

            rows.append(
                (
                    hemi,
                    mode,
                    float(
                        beta_star[
                            mode
                        ]
                    ),
                )
            )

    return rows


# ============================================================
# RUN NULL
#
# Independent deterministic RNG per permutation, exactly as
# historical generator:
#     np.random.default_rng(SEED + perm)
# ============================================================

null_rows = []


for perm in range(
    N_PERM
):

    rng = np.random.default_rng(
        SEED + perm
    )

    shuffled = {}

    for stim_run in range(
        1,
        10,
    ):

        shuffled[
            stim_run
        ] = rng.permutation(
            event_by_run[
                stim_run
            ][
                "weights"
            ]
        )


    perm_beta_rows = []


    for (
        subject,
        run
    ), cache in perm_cache.items():

        stim_run = cache[
            "stimulus_run"
        ]

        ev = event_by_run[
            stim_run
        ]

        g_full = build_shift_regressor(
            cache[
                "T_full"
            ],
            ev[
                "onsets"
            ],
            shuffled[
                stim_run
            ],
        )

        fitted = fit_permuted_shift(
            g_full,
            cache,
        )

        for (
            hemi,
            mode,
            beta_star
        ) in fitted:

            perm_beta_rows.append({
                "subject":
                    subject,

                "run":
                    run,

                "hemisphere":
                    hemi,

                "mode":
                    mode,

                "beta_star":
                    beta_star,
            })


    perm_df = pd.DataFrame(
        perm_beta_rows
    )

    profile = aggregate_group_profile(
        perm_df,
        "beta_star",
    )

    null_mean = float(
        np.mean(
            profile
        )
    )

    null_mean_abs = float(
        np.mean(
            np.abs(
                profile
            )
        )
    )

    null_rms = float(
        np.sqrt(
            np.mean(
                profile ** 2
            )
        )
    )


    null_rows.append({
        "perm":
            perm,

        "mean_beta_star":
            null_mean,

        "mean_abs_beta_star":
            null_mean_abs,

        "profile_rms":
            null_rms,
    })


    print(
        f"perm {perm + 1:5d}/{N_PERM}: "
        f"mean={null_mean:+.6f}, "
        f"meanabs={null_mean_abs:.6f}, "
        f"rms={null_rms:.6f}"
    )


null_df = pd.DataFrame(
    null_rows
)


# ============================================================
# P VALUES
#
# Historical primary statistic:
# signed mean beta*
#
# Because empirical shift is positive, use upper tail.
# Mean-absolute and RMS are supplementary two-sided-magnitude
# style summaries, assessed by upper tail.
# ============================================================

def upper_tail_p(
    null,
    empirical,
):

    null = np.asarray(
        null,
        float,
    )

    return (
        np.sum(
            null >= empirical
        )
        + 1
    ) / (
        len(null)
        + 1
    )


p_mean = upper_tail_p(
    null_df[
        "mean_beta_star"
    ],
    emp_mean,
)

p_abs = upper_tail_p(
    null_df[
        "mean_abs_beta_star"
    ],
    emp_mean_abs,
)

p_rms = upper_tail_p(
    null_df[
        "profile_rms"
    ],
    emp_rms,
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []

for stat, empirical, p in [
    (
        "mean_beta_star",
        emp_mean,
        p_mean,
    ),
    (
        "mean_abs_beta_star",
        emp_mean_abs,
        p_abs,
    ),
    (
        "profile_rms",
        emp_rms,
        p_rms,
    ),
]:

    vals = null_df[
        stat
    ].to_numpy(float)

    summary_rows.append({
        "statistic":
            stat,

        "empirical":
            empirical,

        "null_mean":
            float(
                np.mean(
                    vals
                )
            ),

        "null_sd":
            float(
                np.std(
                    vals,
                    ddof=1,
                )
            )
            if len(vals) > 1
            else np.nan,

        "null_2.5%":
            float(
                np.percentile(
                    vals,
                    2.5,
                )
            ),

        "null_97.5%":
            float(
                np.percentile(
                    vals,
                    97.5,
                )
            ),

        "p_upper":
            p,

        "N_PERM":
            N_PERM,
    })


summary = pd.DataFrame(
    summary_rows
)


print(
    "\n" + "=" * 80
)

print(
    "CORRECTED SENTENCE-SHIFT SHUFFLE NULL"
)

print(
    "=" * 80
)

print(
    summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.10f}"
    )
)

if N_PERM < 10000:

    print(
        "\nIMPORTANT: these p-values are QC-only "
        f"because N_PERM={N_PERM}. "
        "Do not interpret them inferentially."
    )


# ============================================================
# SAVE
# ============================================================

null_file = (
    OUT_DIR
    / f"sentence_shift_shuffle_null_corrected_{N_PERM}.csv"
)

summary_file = (
    OUT_DIR
    / f"sentence_shift_shuffle_null_corrected_{N_PERM}_summary.csv"
)

mapping_file = (
    OUT_DIR
    / "sentence_shift_stimulus_mapping_QC.csv"
)

null_df.to_csv(
    null_file,
    index=False,
)

summary.to_csv(
    summary_file,
    index=False,
)

mapping_df.to_csv(
    mapping_file,
    index=False,
)


print(
    "\nSaved:"
)

print(
    null_file
)

print(
    summary_file
)

print(
    mapping_file
)

print(
    "\nAll mandatory empirical QC stages passed."
)
