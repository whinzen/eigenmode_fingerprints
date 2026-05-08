from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

BASE = Path.home() / "eigenmode_fingerprints"
GRP = BASE / "pang_out" / "group"

ENERGY_ALL = GRP / "energy_all.csv"
GROUP_SPEC = GRP / "energy_spectrum_group.csv"

OUT_DIR = GRP / "criticality_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# 1. FIT STABILITY
# -----------------------

FIT_RANGES = [
    (1, 40),
    (1, 60),
    (10, 80),
    (20, 100),
    (40, 120),
]


def fit_range(df, kmin, kmax):
    d = df[(df["mode_k"] >= kmin) & (df["mode_k"] <= kmax)].copy()
    d = d[(d["lam"] > 0) & (d["Emean"] > 0)]

    x = np.log10(d["lam"].values)
    y = np.log10(d["Emean"].values)

    slope, intercept, r, p, stderr = linregress(x, y)

    return {
        "kmin": kmin,
        "kmax": kmax,
        "slope": slope,
        "r2": r**2,
        "n": len(d),
    }


def run_fit_stability():
    df = pd.read_csv(GROUP_SPEC)

    results = []
    for kmin, kmax in FIT_RANGES:
        res = fit_range(df, kmin, kmax)
        results.append(res)

    out = pd.DataFrame(results)
    out.to_csv(OUT_DIR / "fit_stability.csv", index=False)

    print("\n=== FIT STABILITY ===")
    print(out)

    return out


# -----------------------
# 2. CURVATURE TEST
# -----------------------

def curvature_test():
    df = pd.read_csv(GROUP_SPEC)

    # exclude mode 0
    df = df[df["mode_k"] >= 1].copy()
    df = df[(df["lam"] > 0) & (df["Emean"] > 0)]

    x = np.log10(df["lam"].values)
    y = np.log10(df["Emean"].values)

    # linear fit
    lin = np.polyfit(x, y, 1)

    # quadratic fit
    quad = np.polyfit(x, y, 2)

    print("\n=== CURVATURE TEST ===")
    print(f"Linear slope: {lin[0]:.4f}")
    print(f"Quadratic term: {quad[0]:.6f}")

    # plot
    xs = np.linspace(x.min(), x.max(), 200)
    y_lin = np.polyval(lin, xs)
    y_quad = np.polyval(quad, xs)

    plt.figure(figsize=(5, 4))
    plt.plot(x, y, "o", alpha=0.4, label="data")
    plt.plot(xs, y_lin, label="linear fit")
    plt.plot(xs, y_quad, "--", label="quadratic fit")

    plt.xlabel("log10 λ")
    plt.ylabel("log10 E")
    plt.title("Curvature test (log–log space)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUT_DIR / "curvature_test.png", dpi=300)
    plt.close()

    return quad[0]


# -----------------------
# 3. SUBJECT VARIABILITY
# -----------------------

def subject_variability():
    df = pd.read_csv(ENERGY_ALL)

    slopes = []

    for (sub, run, hemi), d in df.groupby(["subject", "run", "hemi"]):
        d = d[(d["mode_k"] >= 1) & (d["mode_k"] <= 60)]
        d = d[(d["lam"] > 0) & (d["E"] > 0)]

        if len(d) < 10:
            continue

        x = np.log10(d["lam"].values)
        y = np.log10(d["E"].values)

        slope, _, r, _, _ = linregress(x, y)

        slopes.append({
            "subject": sub,
            "run": run,
            "hemi": hemi,
            "slope": slope,
            "r": r,
        })

    df_slopes = pd.DataFrame(slopes)
    df_slopes.to_csv(OUT_DIR / "subject_slopes.csv", index=False)

    print("\n=== SUBJECT VARIABILITY ===")
    print(df_slopes["slope"].describe())

    # histogram
    plt.figure(figsize=(5, 4))
    plt.hist(df_slopes["slope"], bins=30)
    plt.xlabel("Slope")
    plt.ylabel("Count")
    plt.title("Distribution of power-law slopes")
    plt.tight_layout()

    plt.savefig(OUT_DIR / "slope_histogram.png", dpi=300)
    plt.close()

    return df_slopes


# -----------------------
# MAIN
# -----------------------

def main():
    fit_df = run_fit_stability()
    curvature = curvature_test()
    slopes_df = subject_variability()

    print("\n=== SUMMARY ===")
    print(f"Mean slope (subject-level): {slopes_df['slope'].mean():.4f}")
    print(f"Std slope  (subject-level): {slopes_df['slope'].std():.4f}")
    print(f"Curvature (quadratic term): {curvature:.6f}")


if __name__ == "__main__":
    main()