import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import nibabel as nib

# ---------------- Paths ----------------
BASE = Path.home() / "eigenmode_fingerprints"
FUNC_DIR = BASE / "data/empirical"
REGRESSOR_INPUT_DIR = BASE / "pang_out/regressors/boundary_vp_hrf"
REGRESSOR_OUTPUT_DIR = BASE / "pang_out/regressors/boundary_vp_hrf_per_subject"
CSV_PATH = BASE / "ds003643/annotation/EN/repunct/lppEN.csv"

REGRESSOR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load stimulus metadata with [word, onset, offset, run_id]
stimulus_df = pd.read_csv(CSV_PATH)
stimulus_df["run_id"] = stimulus_df["run_id"].astype(int)  # should be 1–9
# --------------------------------------

# --- Gather subject list ---
subjects = sorted([p.name for p in FUNC_DIR.glob("sub-*")])
valid_subjects = [s for s in subjects if len(list((FUNC_DIR / s / "func").glob("*_bold.func.gii"))) >= 9]
print(f"Found {len(valid_subjects)} subjects with ≥9 runs")

# --------- Main loop ---------
for subj in tqdm(valid_subjects, desc="Processing subjects"):
    subj_dir = FUNC_DIR / subj / "func"
    func_files = sorted(subj_dir.glob("*_bold.func.gii"))

    # Extract actual fMRI run-IDs from filenames and sort
    run_files = {}
    for f in func_files:
        try:
            run = int(f.name.split("_run-")[1].split("_")[0])
            run_files[run] = f
        except Exception:
            continue

    if len(run_files) != 9:
        print(f"⚠️ Skipping {subj} — only found {len(run_files)} functional runs")
        continue

    # Re-map to stimulus run_ids (1–9)
    ordered_runs = dict(enumerate(sorted(run_files.items()), start=1))  # {1: (6, Path(...)), ...}

    for stim_run_id, (fmri_run_id, func_file) in ordered_runs.items():
        try:
            n_trs = len(nib.load(func_file).darrays)
        except Exception as e:
            print(f"❌ Error in {subj} run {stim_run_id}: {e}")
            continue

        # Load canonical stimulus regressor for this run_id (1-based)
        reg_path = REGRESSOR_INPUT_DIR / f"run-{stim_run_id}.npy"
        if not reg_path.exists():
            print(f"⚠️ Missing canonical regressor for run-{stim_run_id}")
            continue

        X = np.load(reg_path)

        # Adjust to match subject's actual TR count
        if len(X) > n_trs:
            X = X[:n_trs]
        elif len(X) < n_trs:
            X = np.pad(X, (0, n_trs - len(X)))

        # Save subject-specific version
        out_path = REGRESSOR_OUTPUT_DIR / f"{subj}_run-{fmri_run_id}.npy"
        np.save(out_path, X)

print("✅ Done generating subject-specific HRF regressors.")