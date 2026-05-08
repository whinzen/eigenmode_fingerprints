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
FUNC_DIR = BASE / "data" / "empirical"
TRANS_CSV = BASE / "pang_out" / "word_transition_geometry" / "word_transition_geometry.csv"
OUT_BASE = BASE / "pang_out" / "regressors"

TR = 2.0
METRICS = [
    "shift",
    "pred_error_ar",
    "pred_error_subspace",
    "curvature",
    # "pred_error_lin",   # optional
]

# =========================
# Helpers
# =========================
def get_n_trs_from_gifti(func_file: Path) -> int:
    img = nib.load(str(func_file))
    # each darray is one timepoint
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
        out[m] = (xm ** (a - 1) * np.exp(-xm / scale)) / (math.gamma(a) * scale ** a)
        return out

    h = gamma_pdf(t, 6) - 0.5 * gamma_pdf(t, 16)
    if np.max(np.abs(h)) > 0:
        h = h / np.max(np.abs(h))
    return h

HRF = canonical_hrf(TR)

def convolve_hrf(x, hrf):
    y = np.convolve(x, hrf, mode="full")[: len(x)]
    return y

def zscore_safe(x):
    x = np.asarray(x, float)
    if not np.all(np.isfinite(x)):
        return x
    s = x.std()
    if s == 0:
        return x - x.mean()
    return (x - x.mean()) / s

def get_subject_run_mapping(func_subj_dir: Path):
    """
    Map annotation run ids 1..N to actual subject run numbers by sorting subject runs.
    """
    run_files = sorted(func_subj_dir.glob("*_hemi-L_space-fsaverage5_bold.func.gii"))
    if len(run_files) == 0:
        return {}

    def parse_run(p):
        m = re.search(r"_run-(\d+)_", p.name)
        return int(m.group(1))

    actual_runs = sorted(parse_run(p) for p in run_files)
    return {i + 1: run for i, run in enumerate(actual_runs)}

def build_impulse_regressor(onsets_sec, values, n_trs, tr=2.0):
    x = np.zeros(n_trs, dtype=float)
    for onset, val in zip(onsets_sec, values):
        if not np.isfinite(onset) or not np.isfinite(val):
            continue
        idx = int(round(onset / tr))
        if 0 <= idx < n_trs:
            x[idx] += float(val)
    return x

# =========================
# Main
# =========================
def main():
    if not TRANS_CSV.exists():
        raise FileNotFoundError(f"Missing transition table: {TRANS_CSV}")

    trans = pd.read_csv(TRANS_CSV)

    required = {"run_id", "onset", "boundary_flag", *METRICS}
    missing = required.difference(trans.columns)
    if missing:
        raise RuntimeError(f"Missing columns in transition table: {sorted(missing)}")

    # Keep only within-run transitions
    if "same_run" in trans.columns:
        trans = trans[trans["same_run"] == 1].copy()

    subjects = sorted([p.name for p in FUNC_DIR.glob("sub-*") if p.is_dir()])
    print(f"Found {len(subjects)} subjects")

    for metric in METRICS:
        out_dir = OUT_BASE / f"{metric}_hrf_per_subject"
        out_dir.mkdir(parents=True, exist_ok=True)

    for subj in tqdm(subjects, desc="Subjects"):
        func_subj_dir = FUNC_DIR / subj / "func"
        if not func_subj_dir.exists():
            continue

        mapping = get_subject_run_mapping(func_subj_dir)
        if not mapping:
            continue

        # Loop over annotation runs (1..N available for that subject)
        for ann_run_id, actual_run in mapping.items():
            # locate a left-hemi file just to get TR length
            func_file = next(iter(sorted(func_subj_dir.glob(f"*_run-{actual_run}_hemi-L_space-fsaverage5_bold.func.gii"))), None)
            if func_file is None:
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
                    x = build_impulse_regressor(onsets[good], vals[good], n_trs, tr=TR)
                    x = convolve_hrf(x, HRF)
                    x = zscore_safe(x)

                out_path = OUT_BASE / f"{metric}_hrf_per_subject" / f"{subj}_run-{actual_run}.npy"
                np.save(out_path, x)

    print("✅ Done building subjectwise transition-metric regressors.")

if __name__ == "__main__":
    main()