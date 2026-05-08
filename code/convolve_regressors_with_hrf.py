import os
import numpy as np
from scipy.stats import gamma
from scipy.signal import fftconvolve

# Set paths on HPC
TR = 2.0  # seconds
OVERSAMPLING = 16
BASE = "/gpfs/home/whinzen/eigenmode_fingerprints/pang_out/regressors"
IN_NP = os.path.join(BASE, "boundary_np")
IN_VP = os.path.join(BASE, "boundary_vp")
OUT_NP = os.path.join(BASE, "boundary_np_hrf")
OUT_VP = os.path.join(BASE, "boundary_vp_hrf")

os.makedirs(OUT_NP, exist_ok=True)
os.makedirs(OUT_VP, exist_ok=True)

# SPM-style HRF
def spm_hrf(tr, oversampling=16, time_length=32):
    dt = tr / oversampling
    time = np.linspace(0, time_length, int(time_length / dt))
    hrf = gamma.pdf(time, 6) - 0.35 * gamma.pdf(time, 12)
    return hrf / hrf.sum()

hrf = spm_hrf(TR, OVERSAMPLING)

# Convolve binary regressor with HRF

def convolve_regressor(binary, hrf, oversampling):
    upsampled = np.repeat(binary, oversampling)
    conv = fftconvolve(upsampled, hrf)[:len(upsampled)]
    downsampled = conv.reshape(-1, oversampling).mean(axis=1)
    return downsampled[:len(binary)]

# Process all files in a directory
def process_all(in_dir, out_dir):
    for fname in sorted(os.listdir(in_dir)):
        if fname.endswith(".npy") and fname.startswith("run-"):
            arr = np.load(os.path.join(in_dir, fname))
            out = convolve_regressor(arr, hrf, OVERSAMPLING)
            np.save(os.path.join(out_dir, fname), out)
            print(f"✓ Saved HRF convolved: {fname} → {out_dir}")

process_all(IN_NP, OUT_NP)
process_all(IN_VP, OUT_VP)
print("✅ All regressors convolved with HRF and saved.")