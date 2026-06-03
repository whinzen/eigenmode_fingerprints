#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

IN = (
    PANG
    / "subcortex"
    / "vta_cortical_eigenmode_comparison"
    / "boundary_vta_cortical_eigenmode_profile_merged.csv"
)

OUT = PANG / "subcortex" / "brainstem_vta_hipp_figures"
OUT.mkdir(parents=True, exist_ok=True)

N_TOP = 20

df = pd.read_csv(IN)

# Exclude global mode
df = df[df["mode_k"] > 0].copy()

df["abs_boundary"] = np.abs(df["boundary_beta"])
df["abs_vta"] = np.abs(df["vta_beta"])

top_boundary = set(df.nlargest(N_TOP, "abs_boundary")["mode_k"])
top_vta = set(df.nlargest(N_TOP, "abs_vta")["mode_k"])
overlap = top_boundary & top_vta

# Overlap stats from your permutation result
jaccard = len(overlap) / len(top_boundary | top_vta)
perm_p = 1e-4

x = df["mode_k"].values
y = df["vta_beta"].values
sem = df["vta_beta_sem"].values

fig, ax = plt.subplots(figsize=(6.4, 3.8))

ax.plot(x, y, linewidth=2.0)
ax.fill_between(x, y - sem, y + sem, alpha=0.25)
ax.axhline(0, color="black", linewidth=1)

# Highlight overlapping top-ranked modes
m = df["mode_k"].isin(overlap)
ax.scatter(
    df.loc[m, "mode_k"],
    df.loc[m, "vta_beta"],
    s=46,
    zorder=5,
    edgecolor="black",
    linewidth=0.7,
    label="Boundary-sensitive & VTA-coupled",
)

ax.text(
    0.98,
    0.95,
    f"Top-20 overlap: {len(overlap)}/20\n"
    f"Jaccard = {jaccard:.2f}\n"
    f"$p_{{perm}}$ = {perm_p:.0e}",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=10,
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor="0.6",
        alpha=0.95,
    ),
)

ax.set_title(
    "VTA coupling targets boundary-sensitive eigenmodes",
    fontsize=12,
    fontweight="bold",
)
ax.set_xlabel("Cortical eigenmode k")
ax.set_ylabel("Residualized VTA-coupling β")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(frameon=False, fontsize=8, loc="lower right")

fig.tight_layout()

png = OUT / "vta_boundary_overlap_single_panel.png"
pdf = OUT / "vta_boundary_overlap_single_panel.pdf"

fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

summary = OUT / "vta_boundary_overlap_single_panel_summary.txt"
summary.write_text(
    f"N_TOP={N_TOP}\n"
    f"top_boundary={sorted(top_boundary)}\n"
    f"top_vta={sorted(top_vta)}\n"
    f"overlap={sorted(overlap)}\n"
    f"intersection={len(overlap)}\n"
    f"union={len(top_boundary | top_vta)}\n"
    f"jaccard={jaccard}\n"
    f"perm_p={perm_p}\n"
)

print(f"Wrote {png}")
print(f"Wrote {pdf}")
print(f"Wrote {summary}")
print("Overlap modes:", sorted(overlap))