#!/usr/bin/env python

import numpy as np
import pandas as pd
from pathlib import Path
import json
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.pyplot as plt

# --- Paths ---
BASE = Path("/home/whinzen/eigenmode_fingerprints/pang_out")
GLM_DIR = BASE / "glm_sentence" / "per_run" / "wordrate"
LAMBDAS_PATH = BASE / "lambdas.npy"
FIG_DIR = BASE / "figures" / "wordrate"
STATS_DIR = BASE / "stats" / "wordrate"

FIG_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUT = STATS_DIR / "wordrate_betas_stats.csv"
PLOT_OUT = FIG_DIR / "wordrate_betas_plot.png"
BARPLOT_OUT = FIG_DIR / "wordrate_sigmode_barplot.png"

K = 119  # Analyze 119 nonzero modes (exclude mode 0)

def load_betas(hemi):
    hemi_dir = GLM_DIR / hemi
    betas = []
    for run_file in sorted(hemi_dir.glob("run-*.json")):
        print(f"📂 Loading {run_file.name}")
        with open(run_file) as f:
            data = json.load(f)
        if "beta" in data:
            beta = np.array(data["beta"]).flatten()
            if len(beta) != 120:
                print(f"⚠️ {run_file.name}: Expected 120 βs, found {len(beta)}")
                continue
            beta = beta[1:]  # 👈 Exclude mode 0
            betas.append(beta)
    return np.stack(betas) if betas else np.array([])

def bootstrap_ci(data, n_boot=5000, ci=95):
    try:
        n = data.shape[0]
        boot_means = np.array([
            np.mean(data[np.random.choice(n, n, replace=True)])
            for _ in range(n_boot)
        ])
        lower = np.percentile(boot_means, (100 - ci) / 2)
        upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
        return lower, upper
    except Exception as e:
        print(f"❌ bootstrap_ci failed: {e}")
        return np.nan, np.nan

def permutation_pval(data, n_perm=5000):
    try:
        obs_mean = np.mean(data)
        null_dist = np.array([
            np.mean(data * np.random.choice([-1, 1], size=len(data)))
            for _ in range(n_perm)
        ])
        p = np.mean(np.abs(null_dist) >= abs(obs_mean))
        return p
    except Exception as e:
        print(f"❌ permutation_pval failed: {e}")
        return np.nan

def main():
    lambdas = np.load(LAMBDAS_PATH)[1:K+1]
    loglam = np.log10(lambdas)

    betasL = load_betas("L")
    betasR = load_betas("R")

    if betasL.shape[0] == 0 or betasR.shape[0] == 0:
        print(f"\n❌ No βs found — check GLM output paths.\nExpected in: {GLM_DIR / 'L'} and {GLM_DIR / 'R'}")
        return

    tL, pL = ttest_1samp(betasL, popmean=0, axis=0)
    tR, pR = ttest_1samp(betasR, popmean=0, axis=0)

    fdrL, sigL = fdrcorrection(pL, alpha=0.05)
    fdrR, sigR = fdrcorrection(pR, alpha=0.05)

    # Defensive bootstrapping
    boot_ci_L = np.array([bootstrap_ci(betasL[:, i]) for i in range(K)])
    boot_ci_R = np.array([bootstrap_ci(betasR[:, i]) for i in range(K)])
    bootL_lo, bootL_hi = boot_ci_L[:, 0], boot_ci_L[:, 1]
    bootR_lo, bootR_hi = boot_ci_R[:, 0], boot_ci_R[:, 1]

    # Defensive permutation
    perm_p_L = np.array([permutation_pval(betasL[:, i]) for i in range(K)])
    perm_p_R = np.array([permutation_pval(betasR[:, i]) for i in range(K)])
    sig_perm_L = perm_p_L < 0.05
    sig_perm_R = perm_p_R < 0.05

    # Validate all arrays have length K
    expected_len = K
    arrays = {
        "lambdas": lambdas,
        "loglam": loglam,
        "beta_L": betasL.mean(axis=0),
        "sem_L": betasL.std(axis=0) / np.sqrt(betasL.shape[0]),
        "t_L": tL, "p_L": pL, "fdr_L": fdrL, "sig_L": sigL,
        "perm_p_L": perm_p_L, "sig_perm_L": sig_perm_L,
        "boot_L_lo": bootL_lo, "boot_L_hi": bootL_hi,
        "beta_R": betasR.mean(axis=0),
        "sem_R": betasR.std(axis=0) / np.sqrt(betasR.shape[0]),
        "t_R": tR, "p_R": pR, "fdr_R": fdrR, "sig_R": sigR,
        "perm_p_R": perm_p_R, "sig_perm_R": sig_perm_R,
        "boot_R_lo": bootR_lo, "boot_R_hi": bootR_hi
    }

    bad = [(k, len(v)) for k, v in arrays.items() if len(v) != expected_len]
    if bad:
        print("❌ Mismatched lengths:")
        for k, l in bad:
            print(f"  {k}: {l}")
        return

    # Build DataFrame
    df = pd.DataFrame({**arrays, "mode_index": np.arange(1, K+1)})
    df.to_csv(CSV_OUT, index=False)
    print(f"✅ Saved stats CSV: {CSV_OUT}")

     # Plot
    plt.figure(figsize=(10, 6))
    sigL = df["sig_L"].values.astype(bool)
    sigR = df["sig_R"].values.astype(bool)

    plt.plot(loglam, df["beta_L"], label="Left", color="blue")
    plt.fill_between(loglam, df["boot_L_lo"], df["boot_L_hi"], color="blue", alpha=0.2)
    plt.plot(loglam, df["beta_R"], label="Right", color="red")
    plt.fill_between(loglam, df["boot_R_lo"], df["boot_R_hi"], color="red", alpha=0.2)
    plt.scatter(loglam[sigL], df["beta_L"].values[sigL], color="blue", s=15, label="FDR p<.05 (L)")
    plt.scatter(loglam[sigR], df["beta_R"].values[sigR], color="red", s=15, label="FDR p<.05 (R)")
    plt.axhline(0, color="gray", linestyle="--")
    plt.xlabel("log₁₀(λ)")
    plt.ylabel("β: Wordrate effect")
    plt.title("Wordrate Effect on Eigenmode Energy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=300)
    print(f"✅ Saved plot PNG: {PLOT_OUT}")
    plt.close()

    # Bar plot of significant counts
     # Bar plot of significant counts
    sig_counts = [
        df["sig_L"].astype(bool).sum(),
        df["sig_R"].astype(bool).sum(),
        df["sig_perm_L"].astype(bool).sum(),
        df["sig_perm_R"].astype(bool).sum()
    ]
    labels = ["FDR L", "FDR R", "Perm L", "Perm R"]
    colors = ["blue", "red", "blue", "red"]
    hatch = ["", "", "//", "//"]
    
    print(df.dtypes)

    plt.figure(figsize=(6, 4))
    for i in range(4):
        plt.bar(i, sig_counts[i], color=colors[i], hatch=hatch[i])
    plt.xticks(range(4), labels)
    plt.ylabel("Number of significant modes")
    plt.title("Significant Wordrate Effects")
    plt.tight_layout()
    plt.savefig(BARPLOT_OUT, dpi=300)
    print(f"✅ Saved bar plot PNG: {BARPLOT_OUT}")
    plt.close()

if __name__ == "__main__":
    main()