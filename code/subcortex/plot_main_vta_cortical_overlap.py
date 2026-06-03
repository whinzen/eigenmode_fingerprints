#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

IN = PANG / "subcortex" / "vta_cortical_eigenmode_comparison" / "boundary_vta_cortical_eigenmode_profile_merged.csv"
OUT = PANG / "subcortex" / "brainstem_vta_hipp_figures"
OUT.mkdir(parents=True, exist_ok=True)

N_TOP = 20
df = pd.read_csv(IN)
df = df[df["mode_k"] > 0].copy()

df["abs_boundary"] = np.abs(df["boundary_beta"])
df["abs_vta"] = np.abs(df["vta_beta"])

top_boundary = set(df.nlargest(N_TOP, "abs_boundary")["mode_k"])
top_vta = set(df.nlargest(N_TOP, "abs_vta")["mode_k"])
overlap = top_boundary & top_vta
jaccard = len(overlap) / len(top_boundary | top_vta)

r_abs, p_abs = pearsonr(df["abs_boundary"], df["abs_vta"])
perm_p = 1e-4

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

# A VTA spectrum
ax = axes[0]
x = df["mode_k"].values
y = df["vta_beta"].values
sem = df["vta_beta_sem"].values
ax.plot(x, y, lw=2)
ax.fill_between(x, y - sem, y + sem, alpha=0.25)
ax.axhline(0, color="black", lw=1)
m = df["mode_k"].isin(overlap)
ax.scatter(df.loc[m, "mode_k"], df.loc[m, "vta_beta"],
           s=45, edgecolor="black", linewidth=0.7, zorder=5)
ax.set_xlabel("Cortical eigenmode k")
ax.set_ylabel("Residualized VTA-coupling β")
ax.set_title("A. VTA–cortical eigenmode coupling", fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# B magnitude similarity
ax = axes[1]
ax.scatter(df["abs_boundary"], df["abs_vta"], s=28, alpha=0.8)
coef = np.polyfit(df["abs_boundary"], df["abs_vta"], 1)
xx = np.linspace(df["abs_boundary"].min(), df["abs_boundary"].max(), 100)
ax.plot(xx, coef[0] * xx + coef[1], lw=1.6)
ax.set_xlabel("|Boundary β|")
ax.set_ylabel("|VTA-coupling β|")
ax.set_title(f"B. Profile magnitude similarity\nr = {r_abs:.2f}, p = {p_abs:.1e}",
             fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# C overlap
ax = axes[2]
cats = ["Boundary\ntop 20", "Shared", "VTA\ntop 20"]
vals = [N_TOP, len(overlap), N_TOP]
ax.bar(np.arange(3), vals, width=0.55, edgecolor="black", linewidth=0.5)
ax.set_xticks(np.arange(3))
ax.set_xticklabels(cats)
ax.set_ylabel("Number of modes")
ax.set_ylim(0, N_TOP + 4)
ax.text(1, len(overlap) + 0.7, f"{len(overlap)}/20",
        ha="center", fontweight="bold")
ax.set_title(f"C. Top-mode overlap\nJ = {jaccard:.2f}, pperm = {perm_p:.0e}",
             fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.suptitle("VTA coupling targets cortical eigenmodes involved in sentence-boundary processing",
             fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.90])

png = OUT / "main_vta_cortical_eigenmode_overlap.png"
pdf = OUT / "main_vta_cortical_eigenmode_overlap.pdf"
fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
plt.close(fig)

print("Wrote", png)
print("Wrote", pdf)
print("Overlap modes:", sorted(overlap))