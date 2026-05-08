#!/usr/bin/env python

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"
OUT = PANG / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

# --- SETTINGS ---
SUB = "sub-EN057"
RUN = 15
HEMI = "L"
K_RECON = 20   # try 10, 20, 40

# paths (adapt if needed)
MODE_PATH = BASE / "modes" / "fsaverage5" / f"{HEMI}_phi.npy"
BOLD_PATH = BASE / "data/empirical" / SUB / "func" / f"{SUB}_task-lppEN_run-{RUN}_hemi-{HEMI}_space-fsaverage5_bold.func.gii"

# --- LOAD ---
phi = np.load(MODE_PATH)  # (V × K)
img = nib.load(str(BOLD_PATH))

# take mean across time or a GLM beta map if you prefer
data = np.vstack([d.data for d in img.darrays]).T  # (V × T)
beta_map = data.mean(axis=1)  # simple example

# --- PROJECT ---
coeffs = phi.T @ beta_map   # (K,)
recon = phi[:, :K_RECON] @ coeffs[:K_RECON]

residual = beta_map - recon

# --- METRICS ---
r = np.corrcoef(beta_map, recon)[0,1]
r2 = 1 - (np.var(residual) / np.var(beta_map))

# --- PLOT ---
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

vmax = np.percentile(np.abs(beta_map), 99)

axes[0].scatter(range(len(beta_map)), beta_map, s=1)
axes[0].set_title("A: Empirical map")

axes[1].scatter(range(len(recon)), recon, s=1)
axes[1].set_title(f"B: Reconstruction (K={K_RECON})\nr={r:.2f}, R²={r2:.2f}")

axes[2].scatter(range(len(residual)), residual, s=1)
axes[2].set_title("C: Residual")

for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()

out = OUT / f"lowpass_reconstruction_{SUB}_run{RUN}_{HEMI}.png"
plt.savefig(out, dpi=300)
plt.close()

print(f"✅ wrote {out}")
print(f"r = {r:.3f}, R² = {r2:.3f}")