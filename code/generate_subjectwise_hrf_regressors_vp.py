import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.signal import fftconvolve
import pandas as pd

# === Parameters ===
TR = 1.5
regressor_type = "boundary_vp"
csv_path = "eigenmode_fingerprints/ds003643/annotation/EN/repunct/lppEN.csv"
func_dir = "eigenmode_fingerprints/data/empirical"
out_dir = f"eigenmode_fingerprints/pang_out/regressors/{regressor_type}_hrf_per_subject"
os.makedirs(out_dir, exist_ok=True)

# === Canonical HRF ===
def spm_hrf(tr, oversampling=16, time_length=32., onset=0.):
    dt = tr / oversampling
    time_stamps = np.arange(0, time_length, dt)
    peak1 = 6
    undershoot = 16
    dispersion1 = 1.
    dispersion2 = 1.
    ratio = 6.
    hrf = (time_stamps / peak1) ** dispersion1 * np.exp(-(time_stamps - peak1) / dispersion1)
    hrf -= (time_stamps / undershoot) ** dispersion2 * np.exp(-(time_stamps - undershoot) / dispersion2) / ratio
    hrf[hrf < 0] = 0
    return hrf / hrf.sum()

def convolve_hrf(x, tr):
    hrf = spm_hrf(tr)
    return fftconvolve(x, hrf, mode='full')[:len(x)]

# === Load sentence-level annotation CSV ===
df = pd.read_csv(csv_path)
df["onset"] = df["onset"].astype(float)
df = df[df["boundary_vp"] == 1]  # Only VP boundaries

# === Map subject IDs to 9-run subset ===
subjects = sorted([d for d in os.listdir(func_dir) if d.startswith("sub-EN")])
print(f"Found {len(subjects)} subjects with ≥9 runs")

for subj in tqdm(subjects, desc="Processing subjects"):
    func_subdir = os.path.join(func_dir, subj, "func")
    func_files = sorted([f for f in os.listdir(func_subdir) if f.endswith(".func.gii")])
    if len(func_files) < 9:
        print(f"⚠️ Skipping {subj} — only found {len(func_files)} functional runs")
        continue

    # Sort and remap run indices to 1–9 based on order in the subject folder
    run_mapping = {}
    for i, f in enumerate(func_files[:9]):
        run_id = int(f.split("run-")[1].split("_")[0])
        run_mapping[i+1] = run_id

    for stim_run_id, fmri_run_id in run_mapping.items():
        bold_file = os.path.join(func_subdir, f"sub-{subj.split('-')[1]}_task-lppEN_run-{fmri_run_id}_hemi-L_space-fsaverage5_bold.func.gii")
        if not os.path.exists(bold_file):
            continue

        try:
            gii = nib.load(bold_file)
            n_trs = gii.darrays[0].data.shape[0]
        except:
            print(f"❌ Failed to load {bold_file}")
            continue

        onsets = df[(df["subject"] == subj) & (df["run_id"] == stim_run_id)]["onset"].values
        if len(onsets) == 0:
            continue

        reg = np.zeros(n_trs)
        for onset in onsets:
            tr_idx = int(np.floor(onset / TR))
            if 0 <= tr_idx < n_trs:
                reg[tr_idx] = 1.0

        reg_hrf = convolve_hrf(reg, TR)
        out_path = os.path.join(out_dir, f"{subj}_run-{fmri_run_id}.npy")
        np.save(out_path, reg_hrf)

print("✅ Done generating subject-specific HRF regressors for VP boundaries.")