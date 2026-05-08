import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import gamma

BASE = Path.home() / "eigenmode_fingerprints"
CSV_PATH = BASE / "ds003643/annotation/EN/repunct/lppEN.csv"
OUT_DIR = BASE / "pang_out/regressors/sentence_boundary_hrf_per_subject"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TR = 2.0

def spm_hrf(tr, length=32.0):
    t = np.arange(0, length, tr)
    peak = gamma.pdf(t, 6)
    undershoot = gamma.pdf(t, 16)
    hrf = peak - 0.5 * undershoot
    return hrf / hrf.max()

HRF = spm_hrf(TR)

df = pd.read_csv(CSV_PATH)
df = df.sort_values(["run_id", "snt_id", "token_id"]).reset_index(drop=True)

df["boundary"] = (df["snt_id"] != df["snt_id"].shift(1)).astype(int)
df.loc[0, "boundary"] = 1

for run, g in df.groupby("run_id"):
    onsets = g.loc[g["boundary"] == 1, "onset"].values
    T = int(np.ceil(g["offset"].max() / TR))

    x = np.zeros(T)
    for t in onsets:
        idx = int(round(t / TR))
        if idx < T:
            x[idx] = 1

    x_hrf = np.convolve(x, HRF)[:T]

    np.save(OUT_DIR / f"run-{run:02d}.npy", x_hrf)

print("✅ Boundary regressors built")