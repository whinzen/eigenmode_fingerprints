#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

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


def main():
    df = pd.read_csv(IN).copy()

    # Exclude global/DC mode.
    df = df[df["mode_k"] > 0].copy()

    df["abs_boundary"] = np.abs(df["boundary_beta"])
    df["abs_vta"] = np.abs(df["vta_beta"])

    r_abs, p_abs = pearsonr(df["abs_boundary"], df["abs_vta"])

    top_boundary = set(df.nlargest(N_TOP, "abs_boundary")["mode_k"])
    top_vta = set(df.nlargest(N_TOP, "abs_vta")["mode_k"])
    overlap = top_boundary & top_vta

    inter = len(overlap)
    union = len(top_boundary | top_vta)
    jaccard = inter / union

    # Permutation test
    rng = np.random.default_rng(1)
    modes = df["mode_k"].values
    null = []

    for _ in range(10000):
        perm = set(rng.choice(modes, size=N_TOP, replace=False))
        i = len(top_boundary & perm)
        u = len(top_boundary | perm)
        null.append(i / u)

    null = np.asarray(null)
    p_perm = (np.sum(null >= jaccard) + 1) / (len(null) + 1)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(17.5, 4.8),
        gridspec_kw={"width_ratios": [1.2, 1.2, 1.0, 0.9]},
    )

    # ------------------------------------------------------------
    # Panel A: boundary profile
    # ------------------------------------------------------------
    ax = axes[0]
    x = df["mode_k"].values
    y = df["boundary_beta"].values

    ax.plot(x, y, linewidth=1.8)
    ax.axhline(0, color="black", linewidth=1)
    ax.scatter(
        df[df["mode_k"].isin(overlap)]["mode_k"],
        df[df["mode_k"].isin(overlap)]["boundary_beta"],
        s=30,
        zorder=3,
        label="overlap",
    )
    ax.set_xlabel("Cortical eigenmode k")
    ax.set_ylabel("Sentence-boundary β")
    ax.set_title("A. Boundary eigenmode profile", fontweight="bold", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ------------------------------------------------------------
    # Panel B: VTA coupling profile
    # ------------------------------------------------------------
    ax = axes[1]
    y = df["vta_beta"].values
    sem = df["vta_beta_sem"].values

    ax.plot(x, y, linewidth=1.8)
    ax.fill_between(x, y - sem, y + sem, alpha=0.25)
    ax.axhline(0, color="black", linewidth=1)
    ax.scatter(
        df[df["mode_k"].isin(overlap)]["mode_k"],
        df[df["mode_k"].isin(overlap)]["vta_beta"],
        s=30,
        zorder=3,
        label="overlap",
    )
    ax.set_xlabel("Cortical eigenmode k")
    ax.set_ylabel("Residualized VTA-coupling β")
    ax.set_title("B. VTA–cortical eigenmode coupling", fontweight="bold", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ------------------------------------------------------------
    # Panel C: absolute profile scatter
    # ------------------------------------------------------------
    ax = axes[2]

    ax.scatter(
        df["abs_boundary"],
        df["abs_vta"],
        s=28,
        alpha=0.75,
    )

    # Regression line
    xx = np.linspace(df["abs_boundary"].min(), df["abs_boundary"].max(), 200)
    slope, intercept = np.polyfit(df["abs_boundary"], df["abs_vta"], 1)
    ax.plot(xx, slope * xx + intercept, linewidth=1.8)

    ax.set_xlabel("|Boundary β|")
    ax.set_ylabel("|VTA-coupling β|")
    ax.set_title(
        f"C. Profile magnitude similarity\nr={r_abs:.2f}, p={p_abs:.1e}",
        fontweight="bold",
        fontsize=11,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ------------------------------------------------------------
    # Panel D: top-mode overlap
    # ------------------------------------------------------------
    ax = axes[3]

    categories = ["Boundary\ntop 20", "Shared", "VTA\ntop 20"]
    values = [N_TOP, inter, N_TOP]

    ax.bar(
        np.arange(3),
        values,
        width=0.55,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel("Number of modes")
    ax.set_ylim(0, N_TOP + 4)
    ax.set_title(
        f"D. Top-mode overlap\nJ={jaccard:.2f}, p={p_perm:.1e}",
        fontweight="bold",
        fontsize=11,
    )

    ax.text(
        1,
        inter + 0.7,
        f"{inter}/20",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(w_pad=1.5)

    out_png = OUT / "boundary_vta_cortical_eigenmode_overlap.png"
    out_pdf = OUT / "boundary_vta_cortical_eigenmode_overlap.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    caption = f"""Figure X. VTA coupling preferentially targets sentence-boundary-sensitive cortical eigenmodes.
(A) Cortical eigenmode profile of sentence-boundary modulation, excluding the global mode. Highlighted points indicate modes shared between the 20 strongest boundary-modulated modes and the 20 strongest VTA-coupled modes.
(B) Residualized VTA–cortical eigenmode coupling profile. For each mode, residualized cortical mode energy was regressed on residualized VTA activity. Shaded region indicates SEM across subjects. Highlighted points indicate modes overlapping with the strongest boundary-modulated modes.
(C) Across modes, the magnitude of sentence-boundary modulation was strongly associated with the magnitude of residualized VTA coupling (Pearson r = {r_abs:.3f}, p = {p_abs:.2e}; mode 0 excluded).
(D) Top-mode overlap analysis. Ten of the 20 most strongly boundary-modulated modes were also among the 20 most strongly VTA-coupled modes (Jaccard = {jaccard:.3f}), exceeding chance overlap estimated by permutation testing (p = {p_perm:.4g}). Together, these results indicate that VTA interactions are preferentially concentrated on the same cortical eigenmodes involved in sentence-boundary processing."""
    caption_file = OUT / "boundary_vta_cortical_eigenmode_overlap_caption.txt"
    caption_file.write_text(caption)

    summary_file = OUT / "boundary_vta_cortical_eigenmode_overlap_summary.txt"
    summary_file.write_text(
        f"n_modes={len(df)}\n"
        f"r_abs={r_abs}\n"
        f"p_abs={p_abs}\n"
        f"intersection={inter}\n"
        f"union={union}\n"
        f"jaccard={jaccard}\n"
        f"perm_p={p_perm}\n"
        f"top_boundary={sorted(top_boundary)}\n"
        f"top_vta={sorted(top_vta)}\n"
        f"overlap={sorted(overlap)}\n"
    )

    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {caption_file}")
    print(f"Wrote {summary_file}")

    print("\nOverlap modes:")
    print(sorted(overlap))
    print(f"\nr_abs={r_abs:.4f}, p={p_abs:.3e}")
    print(f"intersection={inter}, jaccard={jaccard:.3f}, permutation p={p_perm:.4g}")


if __name__ == "__main__":
    main()