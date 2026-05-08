from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

BASE = Path.home() / "eigenmode_fingerprints" / "pang_out"
OUT_DIR = BASE / "group_beta_profile_pca"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Add or remove profiles here
PROFILE_FILES = {
    "boundary": BASE / "group_boundary_glm" / "group_onset_hemi-L_subjectlevel_by_mode_excl_k0.csv",
    "sentence_shift": BASE / "group_sentence_shift_glm" / "group_sentence_shift_hemi-L_by_mode_subject_level.csv",
    "shift": BASE / "group_shift_glm" / "group_shift_hemi-L_by_mode_subject_level.csv",
    "pred_error_ar": BASE / "group_pred_error_ar_glm" / "group_pred_error_ar_hemi-L_by_mode_subject_level.csv",
    "pred_error_subspace": BASE / "group_pred_error_subspace_glm" / "group_pred_error_subspace_hemi-L_by_mode_subject_level.csv",
    "curvature": BASE / "group_curvature_glm" / "group_curvature_hemi-L_by_mode_subject_level.csv",
    "shift__plus__pred_error_ar_resid_shift": BASE / "group_shift__plus__pred_error_ar_resid_glm" / "group_shift__plus__pred_error_ar_resid_shift_hemi-L_by_mode_subject_level.csv",
    "shift__plus__pred_error_ar_resid_extra": BASE / "group_shift__plus__pred_error_ar_resid_glm" / "group_shift__plus__pred_error_ar_resid_pred_error_ar_resid_hemi-L_by_mode_subject_level.csv",
    "shift__plus__pred_error_subspace_resid_shift": BASE / "group_shift__plus__pred_error_subspace_resid_glm" / "group_shift__plus__pred_error_subspace_resid_shift_hemi-L_by_mode_subject_level.csv",
    "shift__plus__pred_error_subspace_resid_extra": BASE / "group_shift__plus__pred_error_subspace_resid_glm" / "group_shift__plus__pred_error_subspace_resid_pred_error_subspace_resid_hemi-L_by_mode_subject_level.csv",
    "shift__plus__curvature_resid_shift": BASE / "group_shift__plus__curvature_resid_glm" / "group_shift__plus__curvature_resid_shift_hemi-L_by_mode_subject_level.csv",
    "shift__plus__curvature_resid_extra": BASE / "group_shift__plus__curvature_resid_glm" / "group_shift__plus__curvature_resid_curvature_resid_hemi-L_by_mode_subject_level.csv",
}

def load_profile(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    rename = {}
    if "k" in df.columns and "mode_k" not in df.columns:
        rename["k"] = "mode_k"
    if "lambda" in df.columns and "lam" not in df.columns:
        rename["lambda"] = "lam"
    if rename:
        df = df.rename(columns=rename)

    need = {"mode_k", "beta_mean"}
    missing = need - set(df.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns: {missing}")

    return df[["mode_k", "beta_mean"]].copy()

def main():
    rows = []
    common_modes = None

    # Load and align all profiles
    for name, path in PROFILE_FILES.items():
        if not path.exists():
            print(f"⚠️ Missing: {name} -> {path}")
            continue
        df = load_profile(path)
        df = df[df["mode_k"] > 0].sort_values("mode_k").reset_index(drop=True)

        modes = tuple(df["mode_k"].tolist())
        if common_modes is None:
            common_modes = set(modes)
        else:
            common_modes = common_modes.intersection(modes)

        rows.append((name, df))

    if not rows:
        raise SystemExit("❌ No profiles loaded.")

    common_modes = sorted(common_modes)
    profile_names = []
    X = []

    for name, df in rows:
        df = df[df["mode_k"].isin(common_modes)].sort_values("mode_k")
        y = df["beta_mean"].values.astype(float)

        # Normalize shape only
        m = np.max(np.abs(y))
        if m > 0:
            y = y / m

        profile_names.append(name)
        X.append(y)

    X = np.vstack(X)  # [n_profiles, n_modes]

    pca = PCA()
    scores = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_

    print("\nExplained variance ratio:")
    for i, v in enumerate(evr[:5], start=1):
        print(f"PC{i}: {v:.6f}")

    # Save EVR
    evr_df = pd.DataFrame({
        "PC": [f"PC{i}" for i in range(1, len(evr) + 1)],
        "explained_variance_ratio": evr
    })
    evr_df.to_csv(OUT_DIR / "beta_profile_pca_explained_variance.csv", index=False)

    # Save scores
    scores_df = pd.DataFrame(scores[:, :3], columns=["PC1", "PC2", "PC3"])
    scores_df.insert(0, "profile", profile_names)
    scores_df.to_csv(OUT_DIR / "beta_profile_pca_scores.csv", index=False)

    # Plot EVR
    fig, ax = plt.subplots(figsize=(5.5, 4))
    n_show = min(10, len(evr))
    ax.bar(range(1, n_show + 1), evr[:n_show])
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA of normalized beta-profile shapes")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "beta_profile_pca_evr.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot PC1 loadings as reconstruction quality check
    fig, ax = plt.subplots(figsize=(7, 4.5))
    modes = np.array(common_modes)
    ax.plot(modes, pca.components_[0], linewidth=2)
    ax.set_xlabel("Mode k")
    ax.set_ylabel("PC1 loading")
    ax.set_title("PC1 of normalized beta-profile shapes")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "beta_profile_pc1_loading.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Scatter in PC space
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(scores[:, 0], scores[:, 1], s=40)
    for name, x, y in zip(profile_names, scores[:, 0], scores[:, 1]):
        ax.text(x, y, name, fontsize=8)
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)" if len(evr) > 1 else "PC2")
    ax.set_title("Profiles in PCA space")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "beta_profile_pca_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\n✅ Wrote outputs to: {OUT_DIR}")

if __name__ == "__main__":
    main()