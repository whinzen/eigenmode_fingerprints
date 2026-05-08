from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# =========================
# Paths
# =========================
BASE = Path.home() / "eigenmode_fingerprints" / "pang_out"
OUT_DIR = BASE / "group_beta_profile_pca"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROFILE_FILES = {
    "Boundary": BASE / "group_boundary_glm" / "group_onset_hemi-L_subjectlevel_by_mode_excl_k0.csv",
    "Sentence shift": BASE / "group_sentence_shift_glm" / "group_sentence_shift_hemi-L_by_mode_subject_level.csv",
    "Shift": BASE / "group_shift_glm" / "group_shift_hemi-L_by_mode_subject_level.csv",
    "AR error": BASE / "group_pred_error_ar_glm" / "group_pred_error_ar_hemi-L_by_mode_subject_level.csv",
    "Subspace exit": BASE / "group_pred_error_subspace_glm" / "group_pred_error_subspace_hemi-L_by_mode_subject_level.csv",
    "Curvature": BASE / "group_curvature_glm" / "group_curvature_hemi-L_by_mode_subject_level.csv",
    "Shift (joint AR)": BASE / "group_shift__plus__pred_error_ar_resid_glm" / "group_shift__plus__pred_error_ar_resid_shift_hemi-L_by_mode_subject_level.csv",
    "AR resid": BASE / "group_shift__plus__pred_error_ar_resid_glm" / "group_shift__plus__pred_error_ar_resid_pred_error_ar_resid_hemi-L_by_mode_subject_level.csv",
    "Shift (joint subspace)": BASE / "group_shift__plus__pred_error_subspace_resid_glm" / "group_shift__plus__pred_error_subspace_resid_shift_hemi-L_by_mode_subject_level.csv",
    "Subspace resid": BASE / "group_shift__plus__pred_error_subspace_resid_glm" / "group_shift__plus__pred_error_subspace_resid_pred_error_subspace_resid_hemi-L_by_mode_subject_level.csv",
    "Shift (joint curv)": BASE / "group_shift__plus__curvature_resid_glm" / "group_shift__plus__curvature_resid_shift_hemi-L_by_mode_subject_level.csv",
    "Curvature resid": BASE / "group_shift__plus__curvature_resid_glm" / "group_shift__plus__curvature_resid_curvature_resid_hemi-L_by_mode_subject_level.csv",
}

def load_profile(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    rename = {}
    if "k" in df.columns and "mode_k" not in df.columns:
        rename["k"] = "mode_k"
    if "beta" in df.columns and "beta_mean" not in df.columns:
        rename["beta"] = "beta_mean"
    if rename:
        df = df.rename(columns=rename)

    required = {"mode_k", "beta_mean"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns: {missing}")

    return df[["mode_k", "beta_mean"]].copy()

def normalize_profile(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    m = np.nanmax(np.abs(y))
    if not np.isfinite(m) or m == 0:
        return y
    return y / m

loaded = []
common_modes = None

for name, path in PROFILE_FILES.items():
    if not path.exists():
        print(f"⚠️ Missing: {name} -> {path}")
        continue

    df = load_profile(path)
    df = df[df["mode_k"] > 0].sort_values("mode_k").reset_index(drop=True)

    modes = set(df["mode_k"].tolist())
    common_modes = modes if common_modes is None else common_modes.intersection(modes)
    loaded.append((name, df))

if not loaded:
    raise SystemExit("❌ No profile files found.")

common_modes = sorted(common_modes)

profile_names = []
X = []

for name, df in loaded:
    df = df[df["mode_k"].isin(common_modes)].sort_values("mode_k")
    y = df["beta_mean"].values.astype(float)
    y = normalize_profile(y)
    profile_names.append(name)
    X.append(y)

X = np.vstack(X)
modes = np.array(common_modes)

pca = PCA()
scores = pca.fit_transform(X)
evr = pca.explained_variance_ratio_
pc1 = pca.components_[0]

print("Explained variance ratio:")
for i, v in enumerate(evr[:10], start=1):
    print(f"PC{i}: {v:.6f}")

pd.DataFrame({
    "PC": [f"PC{i}" for i in range(1, len(evr) + 1)],
    "explained_variance_ratio": evr
}).to_csv(OUT_DIR / "beta_profile_pca_explained_variance.csv", index=False)

pd.DataFrame({
    "profile": profile_names,
    "PC1": scores[:, 0],
    "PC2": scores[:, 1] if scores.shape[1] > 1 else np.nan,
    "PC3": scores[:, 2] if scores.shape[1] > 2 else np.nan,
}).to_csv(OUT_DIR / "beta_profile_pca_scores.csv", index=False)

pd.DataFrame({
    "mode_k": modes,
    "PC1_loading": pc1
}).to_csv(OUT_DIR / "beta_profile_pc1_loading.csv", index=False)

fig, axes = plt.subplots(
    1, 2, figsize=(10.8, 4.4),
    gridspec_kw={"width_ratios": [1.0, 1.45]}
)

# Panel A: Scree
ax = axes[0]
n_show = min(6, len(evr))
ax.bar(np.arange(1, n_show + 1), evr[:n_show], width=0.7)
ax.set_xlabel("Principal component")
ax.set_ylabel("Explained variance ratio")
ax.set_title("A. Scree plot")
ax.set_xticks(np.arange(1, n_show + 1))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(1, evr[0] + 0.02, f"{evr[0]*100:.2f}%", ha="center", va="bottom", fontsize=10)

# Panel B: PC1 loading
ax = axes[1]
ax.plot(modes, pc1, linewidth=2)
ax.axhline(0, color="gray", linewidth=1)
ax.set_xlabel("Eigenmode index (k)")
ax.set_ylabel("PC1 loading")
ax.set_title("B. First principal component of beta-profile shape")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Inset: PC1-PC2 scores (numbered labels to avoid overlap)
# Inset: PC1-PC2 scores using colors instead of text labels
if scores.shape[1] >= 2:
    axins = inset_axes(ax, width="44%", height="52%", loc="upper right", borderpad=1.2)

    color_map = {
        "Boundary": "#7f7f7f",
        "Sentence shift": "#000000",
        "Shift": "#1f77b4",
        "AR error": "#ff7f0e",
        "Subspace exit": "#2ca02c",
        "Curvature": "#d62728",
        "Shift (joint AR)": "#1f77b4",
        "AR resid": "#ff7f0e",
        "Shift (joint subspace)": "#1f77b4",
        "Subspace resid": "#2ca02c",
        "Shift (joint curv)": "#1f77b4",
        "Curvature resid": "#d62728",
    }

    marker_map = {
        "Boundary": "o",
        "Sentence shift": "o",
        "Shift": "o",
        "AR error": "o",
        "Subspace exit": "o",
        "Curvature": "o",
        "Shift (joint AR)": "^",
        "AR resid": "^",
        "Shift (joint subspace)": "^",
        "Subspace resid": "^",
        "Shift (joint curv)": "^",
        "Curvature resid": "^",
    }

    for name, xi, yi in zip(profile_names, scores[:, 0], scores[:, 1]):
        axins.scatter(
            xi, yi,
            s=42,
            marker=marker_map.get(name, "o"),
            facecolor=color_map.get(name, "black"),
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9 if marker_map.get(name, "o") == "o" else 0.75,
            zorder=2
        )

    axins.set_xlabel("PC1", fontsize=7)
    axins.set_ylabel("PC2", fontsize=7)
    axins.tick_params(axis="both", labelsize=6)

    x = scores[:, 0]
    y = scores[:, 1]
    xpad = max(0.002, 0.05 * (x.max() - x.min()))
    yspan = y.max() - y.min()
    ypad = max(0.003, 0.10 * (yspan if yspan > 0 else 1.0))

    axins.set_xlim(x.min() - xpad, x.max() + xpad)
    axins.set_ylim(y.min() - ypad, y.max() + ypad)

    axins.spines["top"].set_visible(False)
    axins.spines["right"].set_visible(False)

    # Compact legend just left of inset
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="#7f7f7f", markeredgecolor='black', markersize=6, label='Boundary'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="#000000", markeredgecolor='black', markersize=6, label='Sentence shift'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="#1f77b4", markeredgecolor='black', markersize=6, label='Shift'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="#ff7f0e", markeredgecolor='black', markersize=6, label='AR error'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="#2ca02c", markeredgecolor='black', markersize=6, label='Subspace exit'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="#d62728", markeredgecolor='black', markersize=6, label='Curvature'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor="white", markeredgecolor='black', markersize=6, label='Residualized'),
    ]

    axins.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=6,
        loc="upper left",
        bbox_to_anchor=(-0.72, 1.02),
        borderaxespad=0.0,
        handletextpad=0.4,
        labelspacing=0.3
    )

    # Tight but readable limits
    xpad = max(0.002, 0.05 * (x.max() - x.min()))
    ypad = max(0.0025, 0.12 * (y.max() - y.min() if y.max() > y.min() else 1.0))

    axins.set_xlim(x.min() - xpad, x.max() + xpad)
    axins.set_ylim(y.min() - ypad, y.max() + ypad)

    axins.spines["top"].set_visible(False)
    axins.spines["right"].set_visible(False)

    # improve visibility when PC2 ~ 0
    axins.set_ylim(
        y.min() - 0.002,
        y.max() + 0.002
    )

    axins.spines["top"].set_visible(False)
    axins.spines["right"].set_visible(False)

fig.suptitle("PCA of normalized eigenmode beta profiles", y=1.02, fontsize=13)
fig.tight_layout()

png = OUT_DIR / "figure_pca_scree_pc1_nature_inset.png"
pdf = OUT_DIR / "figure_pca_scree_pc1_nature_inset.pdf"
svg = OUT_DIR / "figure_pca_scree_pc1_nature_inset.svg"

fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
fig.savefig(svg, bbox_inches="tight")
plt.close(fig)

print(f"✅ wrote {png}")
print(f"✅ wrote {pdf}")
print(f"✅ wrote {svg}")