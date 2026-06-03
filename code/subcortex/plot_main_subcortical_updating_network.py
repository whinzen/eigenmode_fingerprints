#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

OUT = PANG / "subcortex" / "brainstem_vta_hipp_figures"
OUT.mkdir(parents=True, exist_ok=True)

PRED_ORDER = [
    "sentence_boundary",
    "sentence_shift",
    "token_shift",
    "pred_error_ar",
    "pred_error_subspace",
    "curvature",
]

PRED_LABELS = {
    "sentence_boundary": "Boundary",
    "sentence_shift": "Sentence\nshift",
    "token_shift": "Token\nshift",
    "pred_error_ar": "AR pred.\nerror",
    "pred_error_subspace": "Subspace\nexit",
    "curvature": "Curvature",
}

PRED_SHORT = {
    "sentence_boundary": "Boundary",
    "sentence_shift": "Sentence shift",
    "token_shift": "Token shift",
    "pred_error_ar": "AR prediction error",
    "pred_error_subspace": "Subspace exit",
    "curvature": "Curvature",
}

# stable, readable palette
palette = plt.get_cmap("tab10")
PRED_COLORS = {
    p: palette(i) for i, p in enumerate(PRED_ORDER)
}

PAIR_COLORS = {
    "VTA_L–HCL": palette(0),
    "VTA_L–HCR": palette(1),
    "VTA_R–HCL": palette(2),
    "VTA_R–HCR": palette(3),
}

# ---------- load files ----------
hc_f = PANG / "subcortex" / "hipp_mean_signal_group" / "group_all_predictors_hipp_mean_signal_bihemi.csv"
vta_f = PANG / "subcortex" / "brainstem_roi_group" / "group_brainstem_roi_glm_all_predictors.csv"
coupling_f = PANG / "subcortex" / "vta_hipp_coupling_global_residualized" / "vta_hipp_coupling_global_residualized_group.csv"
boundary_coupling_f = PANG / "subcortex" / "vta_hipp_boundary_modulated_coupling" / "boundary_modulated_vta_hipp_coupling_group.csv"
multi_f = PANG / "subcortex" / "vta_multivariate_glm" / "vta_multivariate_glm_bilateral_group.csv"

hc = pd.read_csv(hc_f)
vta = pd.read_csv(vta_f)
coupling = pd.read_csv(coupling_f)
bc = pd.read_csv(boundary_coupling_f)
multi = pd.read_csv(multi_f)

# ---------- harmonize ----------
hc = hc[hc["predictor"].isin(PRED_ORDER)].copy()
hc["predictor"] = pd.Categorical(hc["predictor"], PRED_ORDER, ordered=True)
hc = hc.sort_values("predictor")

vta = vta[(vta["roi"].isin(["VTA_L", "VTA_R"])) & (vta["predictor"].isin(PRED_ORDER))].copy()
vta = (
    vta.groupby("predictor", as_index=False)
    .agg(beta_mean=("beta_mean", "mean"), beta_sem=("beta_sem", "mean"))
)
vta["predictor"] = pd.Categorical(vta["predictor"], PRED_ORDER, ordered=True)
vta = vta.sort_values("predictor")

multi = multi[multi["predictor"].isin(PRED_ORDER)].copy()
multi["predictor"] = pd.Categorical(multi["predictor"], PRED_ORDER, ordered=True)
multi = multi.sort_values("predictor")

# profile similarity
merged = hc[["predictor", "beta_mean"]].merge(
    vta[["predictor", "beta_mean"]],
    on="predictor",
    suffixes=("_hc", "_vta"),
)
r_profile = np.corrcoef(merged["beta_mean_hc"], merged["beta_mean_vta"])[0, 1]

# coupling control summary
coupling_summary = (
    coupling.groupby("coupling_type", as_index=False)
    .agg(
        mean_r=("mean_r_approx", "mean"),
        sem_r=("mean_r_approx", "sem"),
    )
)

ctype_order = ["raw", "global_residualized", "diff_global_residualized"]
ctype_labels = {
    "raw": "Raw",
    "global_residualized": "Global\nresid.",
    "diff_global_residualized": "Diff.\nresid.",
}
coupling_summary["coupling_type"] = pd.Categorical(
    coupling_summary["coupling_type"], ctype_order, ordered=True
)
coupling_summary = coupling_summary.sort_values("coupling_type")

# boundary coupling summary
bc_summary = bc.copy()
bc_summary["pair"] = (
    bc_summary["vta_roi"]
    + "–HC"
    + bc_summary["hipp_hemi"].astype(str)
)
bc_summary["sem_delta_r"] = bc_summary["sem_delta_z"]  # approximate, as Δr is small

# ---------- plot ----------
fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.6))
axes = axes.ravel()

x = np.arange(len(PRED_ORDER))
labels = [PRED_LABELS[p] for p in PRED_ORDER]
bar_colors = [PRED_COLORS[p] for p in PRED_ORDER]

# A hippocampus
ax = axes[0]
ax.bar(
    x,
    hc["beta_mean"],
    yerr=hc["beta_sem"],
    capsize=3,
    color=bar_colors,
    edgecolor="black",
    linewidth=0.4,
)
ax.axhline(0, color="black", lw=1)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_ylabel("β")
ax.set_title("A. Hippocampal responses", fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# B VTA
ax = axes[1]
ax.bar(
    x,
    vta["beta_mean"],
    yerr=vta["beta_sem"],
    capsize=3,
    color=bar_colors,
    edgecolor="black",
    linewidth=0.4,
)
ax.axhline(0, color="black", lw=1)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_ylabel("β")
ax.set_title("B. VTA responses", fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# C profile similarity, no overlapping labels
ax = axes[2]
for _, row in merged.iterrows():
    p = str(row["predictor"])
    ax.scatter(
        row["beta_mean_hc"],
        row["beta_mean_vta"],
        s=65,
        color=PRED_COLORS[p],
        edgecolor="black",
        linewidth=0.5,
        label=PRED_SHORT[p],
    )

coef = np.polyfit(merged["beta_mean_hc"], merged["beta_mean_vta"], 1)
xx = np.linspace(merged["beta_mean_hc"].min(), merged["beta_mean_hc"].max(), 100)
ax.plot(xx, coef[0] * xx + coef[1], color="black", lw=1.5)

ax.set_xlabel("Hippocampus β")
ax.set_ylabel("VTA β")
ax.set_title(f"C. Shared response profile\nr = {r_profile:.2f}", fontweight="bold")
ax.legend(frameon=False, fontsize=7.5, loc="best")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# D coupling controls with SEM
ax = axes[3]
cx = np.arange(len(coupling_summary))
control_colors = [palette(7), palette(8), palette(9)]
ax.bar(
    cx,
    coupling_summary["mean_r"],
    yerr=coupling_summary["sem_r"],
    capsize=3,
    color=control_colors,
    edgecolor="black",
    linewidth=0.4,
)
ax.axhline(0, color="black", lw=1)
ax.set_xticks(cx)
ax.set_xticklabels([ctype_labels[str(c)] for c in coupling_summary["coupling_type"]])
ax.set_ylabel("Mean r")
ax.set_title("D. VTA–hippocampal coupling controls", fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# E boundary modulation with SEM and pair colors
ax = axes[4]
px = np.arange(len(bc_summary))
pair_colors = [PAIR_COLORS.get(p, palette(i)) for i, p in enumerate(bc_summary["pair"])]
ax.bar(
    px,
    bc_summary["mean_delta_r_approx"] if "mean_delta_r_approx" in bc_summary.columns else bc_summary["mean_delta_z"],
    yerr=bc_summary["sem_delta_r"],
    capsize=3,
    color=pair_colors,
    edgecolor="black",
    linewidth=0.4,
)
ax.axhline(0, color="black", lw=1)
ax.set_xticks(px)
ax.set_xticklabels(bc_summary["pair"], rotation=25, ha="right", fontsize=8)
ax.set_ylabel("Δr boundary − random")
ax.set_title("E. Coupling increases at boundaries", fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# F multivariate VTA
ax = axes[5]
ax.bar(
    x,
    multi["beta_mean"],
    yerr=multi["beta_sem"],
    capsize=3,
    color=bar_colors,
    edgecolor="black",
    linewidth=0.4,
)
ax.axhline(0, color="black", lw=1)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_ylabel("Unique β")
ax.set_title("F. Multivariate VTA model", fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.suptitle(
    "Subcortical updating network during naturalistic language comprehension",
    fontsize=16,
    fontweight="bold",
)

fig.tight_layout(rect=[0, 0, 1, 0.95])

png = OUT / "main_subcortical_updating_network.png"
pdf = OUT / "main_subcortical_updating_network.pdf"

fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Wrote", png)
print("Wrote", pdf)