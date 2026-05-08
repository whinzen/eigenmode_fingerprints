# ~/eigenmode_fingerprints/code/plot_by_wavelength.py
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from settings import PANG_OUT, HEMIS

BASE = PANG_OUT / "group_boundary_glm"

def plot_file(fcsv, hemi, out_png):
    df = pd.read_csv(fcsv)
    df = df[(df["lam"] > 0) & np.isfinite(df["beta_mean"]) & np.isfinite(df["beta_sem"])]
    df = df.sort_values("lam")
    lam = df["lam"].to_numpy()
    wl  = 1.0/np.sqrt(lam)  # proxy wavelength
    beta = df["beta_mean"].to_numpy()
    sem  = df["beta_sem"].to_numpy()
    sig  = df["sig_q05"].astype(bool).to_numpy()

    plt.figure(figsize=(7,4))
    plt.plot(wl, beta, lw=2)
    plt.fill_between(wl, beta-sem, beta+sem, alpha=0.2)
    if sig.any():
        plt.scatter(wl[sig], beta[sig], s=18)
    plt.axhline(0, ls="--", lw=1)
    plt.xlabel("Spatial wavelength (1/√λ)")
    plt.ylabel("β (sentence boundary)")
    plt.title(f"Sentence boundary effect vs wavelength – hemi {hemi}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()
    print(f"✅ saved {out_png}")

def main():
    for hemi in HEMIS:
        f = BASE / f"group_onset_hemi-{hemi}_subjectlevel_by_mode_excl_k0.csv"
        out = BASE / f"group_onset_hemi-{hemi}_by_wavelength.png"
        plot_file(f, hemi, out)

if __name__ == "__main__":
    main()