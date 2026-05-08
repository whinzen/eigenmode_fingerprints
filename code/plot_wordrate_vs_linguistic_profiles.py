from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path.home() / "eigenmode_fingerprints" / "pang_out"
OUT_DIR = BASE / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES_SINGLE = {
    "Word rate": BASE / "group_wordrate_glm" / "group_wordrate_by_mode_subject_level.csv",
    "Sentence boundary": BASE / "group_sentence_level_glm" / "group_boundary_by_mode_subject_level.csv",
    "Sentence shift": BASE / "group_sentence_level_glm" / "group_sentence_shift_by_mode_subject_level.csv",
}

FILES_LR = {
    "Token shift": (
        BASE / "group_shift_glm" / "group_shift_hemi-L_by_mode_subject_level.csv",
        BASE / "group_shift_glm" / "group_shift_hemi-R_by_mode_subject_level.csv",
    ),
    "Prediction error (AR)": (
        BASE / "group_pred_error_ar_glm" / "group_pred_error_ar_hemi-L_by_mode_subject_level.csv",
        BASE / "group_pred_error_ar_glm" / "group_pred_error_ar_hemi-R_by_mode_subject_level.csv",
    ),
    "Subspace exit": (
        BASE / "group_pred_error_subspace_glm" / "group_pred_error_subspace_hemi-L_by_mode_subject_level.csv",
        BASE / "group_pred_error_subspace_glm" / "group_pred_error_subspace_hemi-R_by_mode_subject_level.csv",
    ),
    "Curvature": (
        BASE / "group_curvature_glm" / "group_curvature_hemi-L_by_mode_subject_level.csv",
        BASE / "group_curvature_glm" / "group_curvature_hemi-R_by_mode_subject_level.csv",
    ),
}


def load_bihemi_single(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    if "k" in df.columns and "mode_k" not in df.columns:
        df = df.rename(columns={"k": "mode_k"})

    dl = df[df["hemi"] == "L"][["mode_k", "beta_mean", "beta_sem"]].copy()
    dr = df[df["hemi"] == "R"][["mode_k", "beta_mean", "beta_sem"]].copy()

    merged = dl.merge(dr, on="mode_k", suffixes=("_L", "_R"), how="inner")
    merged["beta_mean"] = (merged["beta_mean_L"] + merged["beta_mean_R"]) / 2.0
    merged["beta_sem"] = np.sqrt(
        merged["beta_sem_L"].fillna(0.0) ** 2 +
        merged["beta_sem_R"].fillna(0.0) ** 2
    ) / 2.0

    return merged[["mode_k", "beta_mean", "beta_sem"]].sort_values("mode_k")


def load_bihemi_lr(path_l: Path, path_r: Path) -> pd.DataFrame:
    dl = pd.read_csv(path_l).copy()
    dr = pd.read_csv(path_r).copy()

    if "k" in dl.columns and "mode_k" not in dl.columns:
        dl = dl.rename(columns={"k": "mode_k"})
    if "k" in dr.columns and "mode_k" not in dr.columns:
        dr = dr.rename(columns={"k": "mode_k"})

    merged = dl.merge(dr, on="mode_k", suffixes=("_L", "_R"), how="inner")
    merged["beta_mean"] = (merged["beta_mean_L"] + merged["beta_mean_R"]) / 2.0
    merged["beta_sem"] = np.sqrt(
        merged["beta_sem_L"].fillna(0.0) ** 2 +
        merged["beta_sem_R"].fillna(0.0) ** 2
    ) / 2.0

    return merged[["mode_k", "beta_mean", "beta_sem"]].sort_values("mode_k")


def zscore(v):
    v = np.asarray(v, float)
    s = v.std(ddof=0)
    if s == 0:
        return v - v.mean()
    return (v - v.mean()) / s


def main():
    profiles = {}

    for name, path in FILES_SINGLE.items():
        profiles[name] = load_bihemi_single(path)

    for name, (path_l, path_r) in FILES_LR.items():
        profiles[name] = load_bihemi_lr(path_l, path_r)

    # intersect shared modes and exclude mode 0
    shared = None
    for df in profiles.values():
        s = set(df["mode_k"])
        shared = s if shared is None else (shared & s)
    shared = sorted([k for k in shared if k > 0])

    for name in profiles:
        profiles[name] = profiles[name][profiles[name]["mode_k"].isin(shared)].copy()

    # -------- raw profiles --------
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for name, df in profiles.items():
        ax.plot(df["mode_k"], df["beta_mean"], linewidth=2, label=name)

    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xlabel("Eigenmode index (k)")
    ax.set_ylabel("Mean β")
    ax.set_title("Word rate versus linguistic eigenmode profiles")
    ax.grid(False)
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "wordrate_vs_linguistic_profiles_raw.png", dpi=300, facecolor="white")
    plt.savefig(OUT_DIR / "wordrate_vs_linguistic_profiles_raw.pdf", dpi=300, facecolor="white")
    plt.close()

    # -------- normalized shape comparison --------
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for name, df in profiles.items():
        ax.plot(df["mode_k"], zscore(df["beta_mean"].values), linewidth=2, label=name)

    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xlabel("Eigenmode index (k)")
    ax.set_ylabel("Z-scored β profile")
    ax.set_title("Normalized eigenmode profile comparison")
    ax.grid(False)
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "wordrate_vs_linguistic_profiles_zscore.png", dpi=300, facecolor="white")
    plt.savefig(OUT_DIR / "wordrate_vs_linguistic_profiles_zscore.pdf", dpi=300, facecolor="white")
    plt.close()

    # -------- correlations with word rate --------
    wr = profiles["Word rate"]["beta_mean"].values
    print("\n=== Profile correlations with word rate ===")
    for name, df in profiles.items():
        r = np.corrcoef(wr, df["beta_mean"].values)[0, 1]
        print(f"{name:22s} r = {r:.3f}")


if __name__ == "__main__":
    main()