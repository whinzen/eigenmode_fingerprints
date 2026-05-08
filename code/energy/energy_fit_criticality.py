import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import linregress

# --------- CONFIG ---------
BASE = Path.home() / "eigenmode_fingerprints"
GROUP = BASE / "pang_out" / "group"
FNAME = GROUP / "energy_all.csv"
OUT_FIT = GROUP / "criticality_fits.csv"
FIG_DIR = GROUP / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Fit range in mode space
# Exclude k=0, and avoid fitting the noisiest extreme tail.
KMIN = 1
KMAX = 60
# --------------------------


def fit_powerlaw(df_run: pd.DataFrame, kmin: int = KMIN, kmax: int = KMAX):
    """
    Fit log10(E) ~ log10(lam) within an explicit mode range, excluding mode 0.
    """
    d = df_run.copy()

    # explicit fit window in mode space
    d = d[(d["mode_k"] >= kmin) & (d["mode_k"] <= kmax)].copy()

    # valid values for log-log fit
    d = d[(d["lam"] > 0) & (d["E"] > 0)].copy()

    if len(d) < 3:
        return None, d

    logx = np.log10(d["lam"].values)
    logy = np.log10(d["E"].values)

    slope, intercept, r, p, stderr = linregress(logx, logy)

    res = dict(
        slope=slope,
        intercept=intercept,
        r=r,
        p=p,
        stderr=stderr,
        kmin=int(kmin),
        kmax=int(kmax),
        n_points=int(len(d)),
        lam_min=float(d["lam"].min()),
        lam_max=float(d["lam"].max()),
    )
    return res, d


def plot_spectrum(df_run: pd.DataFrame, df_fit: pd.DataFrame, fit_res: dict, subject: str, run, hemi: str):
    """
    Plot full spectrum plus fitted window and fitted line.
    """
    plt.figure(figsize=(5.8, 4.6))

    # full spectrum
    plt.loglog(
        df_run["lam"],
        df_run["E"],
        "o-",
        alpha=0.6,
        label="Full spectrum"
    )

    # fit window points
    plt.loglog(
        df_fit["lam"],
        df_fit["E"],
        "o-",
        linewidth=2,
        label=f"Fit window (k={fit_res['kmin']}–{fit_res['kmax']})"
    )

    # fitted line on the fit window
    xfit = np.log10(df_fit["lam"].values)
    yfit = fit_res["intercept"] + fit_res["slope"] * xfit
    plt.loglog(
        df_fit["lam"].values,
        10 ** yfit,
        "--",
        linewidth=2,
        label=f"Slope = {fit_res['slope']:.3f}"
    )

    plt.xlabel("Mode eigenvalue λ")
    plt.ylabel("Mean mode energy E")
    plt.title(f"Energy spectrum: {subject} run-{run} {hemi}")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend(frameon=False)

    fname = FIG_DIR / f"spectrum_{subject}_run-{run}_{hemi}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


def analyze_criticality():
    if not FNAME.exists():
        raise FileNotFoundError(f"Missing input file: {FNAME}")

    df = pd.read_csv(FNAME)

    required = {"mode_k", "lam", "E", "subject", "run", "hemi"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in {FNAME}: {missing}")

    fits = []

    for (sub, run, hemi), df_sub in df.groupby(["subject", "run", "hemi"]):
        df_sub = df_sub.sort_values("mode_k").copy()

        fit_res, df_fit = fit_powerlaw(df_sub, kmin=KMIN, kmax=KMAX)

        if fit_res is None:
            print(f"[skip] {sub} run-{run} {hemi}: insufficient valid points in fit window")
            continue

        fit_res.update(subject=sub, run=run, hemi=hemi)
        fits.append(fit_res)

        plot_spectrum(df_sub, df_fit, fit_res, sub, run, hemi)

    if not fits:
        raise RuntimeError("No valid fits were produced.")

    df_fit = pd.DataFrame(fits)
    df_fit.to_csv(OUT_FIT, index=False)

    print(f"✅ Saved criticality fit summary: {OUT_FIT}")
    print(f"✅ Fit range used: mode_k = {KMIN}..{KMAX}")
    print(df_fit[['slope', 'r', 'p', 'stderr']].describe().round(4))


if __name__ == "__main__":
    analyze_criticality()