import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path.home() / "eigenmode_fingerprints" / "pang_out"
NEW_DIR = BASE / "group_curvature_glm"
OUT_DIR = BASE / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# original curvature from your earlier pipeline
ORIG_L = NEW_DIR / "group_curvature_hemi-L_by_mode_subject_level.csv"
ORIG_R = NEW_DIR / "group_curvature_hemi-R_by_mode_subject_level.csv"

NEW_METRICS = {
    "global": "Global curvature",
    "mean": "Mean turning angle",
    "path": "Path length",
    "chord": "Chord length",
}


def load_bihemi_single(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()

    dl = df[df["hemi"] == "L"][["mode_k", "beta_mean"]].copy()
    dr = df[df["hemi"] == "R"][["mode_k", "beta_mean"]].copy()

    merged = dl.merge(dr, on="mode_k", suffixes=("_L", "_R"), how="inner")
    merged["beta"] = (merged["beta_mean_L"] + merged["beta_mean_R"]) / 2.0
    return merged[["mode_k", "beta"]].sort_values("mode_k")


def load_bihemi_lr(path_l: Path, path_r: Path) -> pd.DataFrame:
    dl = pd.read_csv(path_l).copy()
    dr = pd.read_csv(path_r).copy()

    if "k" in dl.columns and "mode_k" not in dl.columns:
        dl = dl.rename(columns={"k": "mode_k"})
    if "k" in dr.columns and "mode_k" not in dr.columns:
        dr = dr.rename(columns={"k": "mode_k"})

    merged = dl.merge(dr, on="mode_k", suffixes=("_L", "_R"), how="inner")
    merged["beta"] = (merged["beta_mean_L"] + merged["beta_mean_R"]) / 2.0
    return merged[["mode_k", "beta"]].sort_values("mode_k")


def zscore(v):
    v = np.asarray(v, float)
    s = v.std(ddof=0)
    if s == 0 or not np.isfinite(s):
        return v - v.mean()
    return (v - v.mean()) / s


def main():
    profiles = {}

    for key, label in NEW_METRICS.items():
        f = NEW_DIR / f"group_{key}_by_mode_subject_level.csv"
        profiles[label] = load_bihemi_single(f)

    orig = load_bihemi_lr(ORIG_L, ORIG_R)
    orig = orig.copy()
    orig["beta"] = -orig["beta"]  # sign-flip for direct visual alignment
    profiles["Original curvature (sign-flipped)"] = orig

    # shared modes, exclude mode 0
    shared = None
    for df in profiles.values():
        s = set(df["mode_k"])
        shared = s if shared is None else (shared & s)
    shared = sorted([k for k in shared if k > 0])

    for name in profiles:
        profiles[name] = profiles[name][profiles[name]["mode_k"].isin(shared)].copy()

    # -------- raw figure --------
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for name, df in profiles.items():
        ax.plot(df["mode_k"], df["beta"], linewidth=2, label=name)

    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xlabel("Eigenmode index (k)")
    ax.set_ylabel("Mean β")
    ax.set_title("Curvature family collapse in eigenmode space")
    ax.grid(False)
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()

    out_png = OUT_DIR / "figure_curvature_family_collapse_raw.png"
    out_pdf = OUT_DIR / "figure_curvature_family_collapse_raw.pdf"
    plt.savefig(out_png, dpi=300, facecolor="white")
    plt.savefig(out_pdf, dpi=300, facecolor="white")
    plt.close()

    # -------- normalized figure --------
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for name, df in profiles.items():
        ax.plot(df["mode_k"], zscore(df["beta"].values), linewidth=2, label=name)

    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xlabel("Eigenmode index (k)")
    ax.set_ylabel("Z-scored β")
    ax.set_title("Normalized curvature family collapse")
    ax.grid(False)
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()

    out_png = OUT_DIR / "figure_curvature_family_collapse_zscore.png"
    out_pdf = OUT_DIR / "figure_curvature_family_collapse_zscore.pdf"
    plt.savefig(out_png, dpi=300, facecolor="white")
    plt.savefig(out_pdf, dpi=300, facecolor="white")
    plt.close()

    # -------- correlation table --------
    names = list(profiles.keys())
    rows = []
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if j <= i:
                continue
            df = profiles[n1].merge(profiles[n2], on="mode_k", suffixes=("_1", "_2"))
            r = np.corrcoef(df["beta_1"], df["beta_2"])[0, 1]
            rows.append({
                "Profile 1": n1,
                "Profile 2": n2,
                "Correlation": r,
            })

    corr_df = pd.DataFrame(rows)
    corr_df["Correlation"] = corr_df["Correlation"].round(4)

    out_csv = BASE / "paper_tables" / "table_curvature_family_collapse_correlations.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    corr_df.to_csv(out_csv, index=False)

    print(f"✅ wrote {OUT_DIR / 'figure_curvature_family_collapse_raw.png'}")
    print(f"✅ wrote {OUT_DIR / 'figure_curvature_family_collapse_raw.pdf'}")
    print(f"✅ wrote {OUT_DIR / 'figure_curvature_family_collapse_zscore.png'}")
    print(f"✅ wrote {OUT_DIR / 'figure_curvature_family_collapse_zscore.pdf'}")
    print(f"✅ wrote {out_csv}\n")

    with pd.option_context("display.max_rows", None, "display.width", 220):
        print(corr_df)


if __name__ == "__main__":
    main()