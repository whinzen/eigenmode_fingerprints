#!/usr/bin/env python

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.signal import welch
from scipy.stats import linregress
from pathlib import Path

# ---------- CONFIG ----------
SUB = "sub-EN113"
RUN = "6"
TR = 1.5  # seconds
FMIN, FMAX = 0.01, 0.15  # Hz for fitting range
BASE = Path.home() / "eigenmode_fingerprints"
FUNC_L = BASE / f"data/empirical/{SUB}/func/{SUB}_task-lppEN_run-{RUN}_hemi-L_space-fsaverage5_bold.func.gii"
FUNC_R = BASE / f"data/empirical/{SUB}/func/{SUB}_task-lppEN_run-{RUN}_hemi-R_space-fsaverage5_bold.func.gii"
SAVE_CSV = BASE / "pang_out/group" / f"{SUB}_run-{RUN}_temporal_scaling.csv"
SAVE_CSV.parent.mkdir(parents=True, exist_ok=True)
# ----------------------------

def load_gii_bold(path):
    gii = nib.load(str(path))
    return np.stack([d.data for d in gii.darrays], axis=1)  # [V, T]

def compute_temporal_scaling(X, tr, fmin=0.01, fmax=0.15):
    Xz = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
    alphas, dfs = [], []
    for x in Xz:
        f, Pxx = welch(x, fs=1/tr, nperseg=min(256, len(x)))
        mask = (f >= fmin) & (f <= fmax) & (Pxx > 0)
        if mask.sum() < 5:
            alphas.append(np.nan)
            dfs.append(np.nan)
            continue
        slope, *_ = linregress(np.log10(f[mask]), np.log10(Pxx[mask]))
        alpha = -slope
        df = (5 - alpha) / 2  # fractal dimension from spectral slope
        alphas.append(alpha)
        dfs.append(df)
    return np.array(alphas), np.array(dfs)

# --- Load & average across hemispheres ---
X_L = load_gii_bold(FUNC_L)
X_R = load_gii_bold(FUNC_R)
X = np.vstack([X_L, X_R])  # [V, T]

# --- Compute per-vertex scaling ---
alpha, Df = compute_temporal_scaling(X, TR, FMIN, FMAX)

# --- Save ---
df_out = pd.DataFrame({
    "vertex": np.arange(len(alpha)),
    "alpha_temporal": alpha,
    "Df_temporal": Df
})
df_out.to_csv(SAVE_CSV, index=False)
print(f"✅ Saved: {SAVE_CSV}")