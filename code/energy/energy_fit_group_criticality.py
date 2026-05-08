from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

BASE = Path.home() / "eigenmode_fingerprints"
GRP = BASE / "pang_out" / "group"
IN_CSV = GRP / "energy_spectrum_group.csv"

# ----------------
# CONFIG
FIT_K_START = 1
FIT_K_END = 60

# plotting range
PLOT_K_START = 1
PLOT_K_END = 80

OUT_CSV = GRP / "group_criticality_fit.csv"

PAPER_DIR = BASE / "pang_out" / "paper_figures"
PAPER_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = PAPER_DIR / "figure_energy_spectrum.png"
OUT_PDF = PAPER_DIR / "figure_energy_spectrum.pdf"
# ----------------


def fit_group_energy(df: pd.DataFrame):
    required = {"mode_k", "lam", "Emean", "Esem"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in {IN_CSV}: {missing}")

    d = df.copy()

    # explicit fit window in mode space; excludes mode 0
    dfit = d[(d["mode_k"] >= FIT_K_START) & (d["mode_k"] <= FIT_K_END)].copy()

    # valid values for log-log fit
    dfit = dfit[(dfit["lam"] > 0) & (dfit["Emean"] > 0)].copy()

    if len(dfit) < 3:
        raise RuntimeError("Not enough valid points in fit window.")

    x = np.log10(dfit["lam"].values)
    y = np.log10(dfit["Emean"].values)

    slope, intercept, r, p, stderr = linregress(x, y)

    fit_info = {
        "slope": slope,
        "intercept": intercept,
        "r": r,
        "p": p,
        "stderr": stderr,
        "kmin": int(FIT_K_START),
        "kmax": int(FIT_K_END),
        "n_points": int(len(dfit)),
        "lam_min": float(dfit["lam"].min()),
        "lam_max": float(dfit["lam"].max()),
        "plot_kmin": int(PLOT_K_START),
        "plot_kmax": int(PLOT_K_END),
    }

    return fit_info, dfit


def plot_group_energy(df: pd.DataFrame, dfit: pd.DataFrame, fit_info: dict):
    dplot = df.copy()

    # exclude mode 0 from plot
    dplot = dplot[dplot["mode_k"] >= PLOT_K_START].copy()
    if PLOT_K_END is not None:
        dplot = dplot[dplot["mode_k"] <= PLOT_K_END].copy()

    dplot = dplot[(dplot["lam"] > 0) & (dplot["Emean"] > 0)].copy()

    lam = dplot["lam"].values
    E = dplot["Emean"].values
    sem = dplot["Esem"].values

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # main spectrum
    ax.loglog(
        lam,
        E,
        "o-",
        lw=2,
        markersize=4,
        label=f"Group mean energy (k={PLOT_K_START}–{PLOT_K_END})"
    )

    ax.fill_between(
        lam,
        np.clip(E - sem, 1e-12, None),
        E + sem,
        alpha=0.2,
        label="SEM"
    )

    # fit window
    ax.loglog(
        dfit["lam"].values,
        dfit["Emean"].values,
        "o-",
        lw=2.2,
        markersize=4,
        label=f"Fit window (k={FIT_K_START}–{FIT_K_END})"
    )

    # fitted line
    lam_fit = np.sort(dfit["lam"].values)
    E_fit_line = 10 ** (
        fit_info["intercept"] + fit_info["slope"] * np.log10(lam_fit)
    )
    ax.loglog(
        lam_fit,
        E_fit_line,
        "k--",
        lw=2,
        label="Power-law fit"
    )

    ax.set_xlabel("Eigenvalue λ")
    ax.set_ylabel("Mean mode energy")
    ax.set_title("Group-level energy spectrum")
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(True)

    ax.legend(frameon=False)

    ax.text(
        0.05, 0.05,
        (
            f"Slope = {fit_info['slope']:.3f}\n"
            f"R² = {fit_info['r']**2:.3f}\n"
            f"Fit k = {fit_info['kmin']}–{fit_info['kmax']}"
        ),
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom"
    )

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300, facecolor="white")
    plt.savefig(OUT_PDF, dpi=300, facecolor="white")
    plt.close()


def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Missing input file: {IN_CSV}")

    df = pd.read_csv(IN_CSV)
    fit_info, dfit = fit_group_energy(df)
    plot_group_energy(df, dfit, fit_info)

    pd.DataFrame([fit_info]).to_csv(OUT_CSV, index=False)

    print("✅ Group-level criticality fit written:")
    print("  ➤", OUT_CSV)
    print("  ➤", OUT_PNG)
    print("  ➤", OUT_PDF)
    print(f"Slope: {fit_info['slope']:.6f}")
    print(f"R²   : {fit_info['r']**2:.6f}")
    print(f"Fit range : mode_k = {fit_info['kmin']}..{fit_info['kmax']}")
    print(f"Plot range: mode_k = {fit_info['plot_kmin']}..{fit_info['plot_kmax']}")
    print(f"n_points  : {fit_info['n_points']}")


if __name__ == "__main__":
    main()