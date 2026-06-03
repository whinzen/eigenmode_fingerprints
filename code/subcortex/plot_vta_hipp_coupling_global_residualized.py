#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

COUPLING_FILE = (
    PANG
    / "subcortex"
    / "vta_hipp_coupling_global_residualized"
    / "vta_hipp_coupling_global_residualized_group.csv"
)

BOUNDARY_FILE = (
    PANG
    / "subcortex"
    / "vta_hipp_boundary_modulated_coupling"
    / "boundary_modulated_vta_hipp_coupling_group.csv"
)

OUT = PANG / "subcortex" / "brainstem_vta_hipp_figures"
OUT.mkdir(parents=True, exist_ok=True)

PAIR_ORDER = [
    ("VTA_L", "L"),
    ("VTA_L", "R"),
    ("VTA_R", "L"),
    ("VTA_R", "R"),
]

PAIR_LABELS = {
    ("VTA_L", "L"): "VTA-L\nHC-L",
    ("VTA_L", "R"): "VTA-L\nHC-R",
    ("VTA_R", "L"): "VTA-R\nHC-L",
    ("VTA_R", "R"): "VTA-R\nHC-R",
}

TYPE_ORDER = [
    "raw",
    "global_residualized",
    "diff_global_residualized",
]

TYPE_LABELS = {
    "raw": "Raw",
    "global_residualized": "Global\nresid.",
    "diff_global_residualized": "Diff +\nglobal\nresid.",
}

COLORS = {
    "raw": "#4C72B0",
    "global_residualized": "#55A868",
    "diff_global_residualized": "#C44E52",
    "boundary": "#8172B2",
}


def stars(p):
    if not np.isfinite(p):
        return ""
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def z_to_r_error(mean_z, sem_z):
    mean_r = np.tanh(mean_z)
    lower = np.tanh(mean_z - sem_z)
    upper = np.tanh(mean_z + sem_z)
    return mean_r, mean_r - lower, upper - mean_r


def load_pair_rows(df, coupling_type):
    d = df[df["coupling_type"] == coupling_type].copy()
    rows = []

    for pair in PAIR_ORDER:
        vta, hc = pair
        hit = d[(d["vta_roi"] == vta) & (d["hipp_hemi"] == hc)]
        if len(hit) != 1:
            raise RuntimeError(f"Missing pair {pair} for coupling_type={coupling_type}")
        rows.append(hit.iloc[0])

    return pd.DataFrame(rows)


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", pad=2)
    ax.margins(x=0.06)


def main():
    coupling = pd.read_csv(COUPLING_FILE)
    boundary = pd.read_csv(BOUNDARY_FILE)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.8, 5.2),
        gridspec_kw={"width_ratios": [1.05, 1.0, 1.05]},
    )

    # ------------------------------------------------------------------
    # Panel A: residualized coupling by pair
    # ------------------------------------------------------------------
    ax = axes[0]

    resid = load_pair_rows(coupling, "global_residualized")

    x = np.arange(len(resid)) * 0.75
    y = resid["mean_r_approx"].values

    yerr_low = []
    yerr_high = []

    for _, row in resid.iterrows():
        _, lo, hi = z_to_r_error(row["mean_z"], row["sem_z"])
        yerr_low.append(lo)
        yerr_high.append(hi)

    yerr = np.vstack([yerr_low, yerr_high])

    ax.bar(
        x,
        y,
        yerr=yerr,
        capsize=4,
        width=0.48,
        color=COLORS["global_residualized"],
        edgecolor="black",
        linewidth=0.5,
    )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [PAIR_LABELS[(row["vta_roi"], row["hipp_hemi"])] for _, row in resid.iterrows()],
        fontsize=8,
    )
    ax.set_ylabel("Residualized coupling r", fontsize=12)
    ax.set_title(
        "A. VTA–hippocampus coupling\nafter global-signal removal",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_ylim(0, 0.58)

    for i, row in enumerate(resid.itertuples()):
        ax.text(
            x[i],
            row.mean_r_approx + 0.035,
            stars(row.p),
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    style_axis(ax)

    # ------------------------------------------------------------------
    # Panel B: raw vs residualized controls averaged across pairs
    # ------------------------------------------------------------------
    ax = axes[1]

    summary_rows = []

    for ctype in TYPE_ORDER:
        d = coupling[coupling["coupling_type"] == ctype].copy()

        mean_z = d["mean_z"].mean()
        sem_z = d["mean_z"].sem()

        mean_r, lo, hi = z_to_r_error(mean_z, sem_z)

        summary_rows.append({
            "coupling_type": ctype,
            "mean_z": mean_z,
            "sem_z": sem_z,
            "mean_r": mean_r,
            "yerr_low": lo,
            "yerr_high": hi,
        })

    summary = pd.DataFrame(summary_rows)

    x = np.arange(len(summary)) * 0.75
    y = summary["mean_r"].values
    yerr = np.vstack([
        summary["yerr_low"].values,
        summary["yerr_high"].values,
    ])

    ax.bar(
        x,
        y,
        yerr=yerr,
        capsize=4,
        width=0.48,
        color=[COLORS[c] for c in summary["coupling_type"]],
        edgecolor="black",
        linewidth=0.5,
    )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [TYPE_LABELS[c] for c in summary["coupling_type"]],
        fontsize=8,
    )
    ax.set_ylabel("Mean coupling r", fontsize=12)
    ax.set_title(
        "B. Coupling control analyses",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_ylim(0, 1.05)

    style_axis(ax)

    # ------------------------------------------------------------------
    # Panel C: boundary modulation
    # ------------------------------------------------------------------
    ax = axes[2]

    rows = []

    for pair in PAIR_ORDER:
        vta, hc = pair
        hit = boundary[(boundary["vta_roi"] == vta) & (boundary["hipp_hemi"] == hc)]
        if len(hit) != 1:
            raise RuntimeError(f"Missing boundary pair {pair}")
        rows.append(hit.iloc[0])

    bplot = pd.DataFrame(rows)

    x = np.arange(len(bplot)) * 0.75
    y = bplot["mean_delta_r_approx"].values

    yerr_low = []
    yerr_high = []

    for _, row in bplot.iterrows():
        _, lo, hi = z_to_r_error(row["mean_delta_z"], row["sem_delta_z"])
        yerr_low.append(lo)
        yerr_high.append(hi)

    yerr = np.vstack([yerr_low, yerr_high])

    ax.bar(
        x,
        y,
        yerr=yerr,
        capsize=4,
        width=0.48,
        color=COLORS["boundary"],
        edgecolor="black",
        linewidth=0.5,
    )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [PAIR_LABELS[(row["vta_roi"], row["hipp_hemi"])] for _, row in bplot.iterrows()],
        fontsize=8,
    )
    ax.set_ylabel("Boundary enhancement Δr", fontsize=12)
    ax.set_title(
        "C. Boundary enhancement of\nVTA–hippocampal coupling",
        fontsize=11,
        fontweight="bold",
    )

    ymax = max(y + yerr[1]) * 1.55
    ax.set_ylim(0, ymax)

    for i, row in enumerate(bplot.itertuples()):
        ax.text(
            x[i],
            row.mean_delta_r_approx + yerr[1][i] + 0.002,
            stars(row.p),
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    style_axis(ax)

    fig.tight_layout(w_pad=0.8)

    out_png = OUT / "vta_hipp_coupling_residualized_boundary_controls.png"
    out_pdf = OUT / "vta_hipp_coupling_residualized_boundary_controls.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    caption = """Figure X. VTA–hippocampus coupling and boundary-related coupling enhancement during naturalistic language comprehension.
(A) Mean VTA–hippocampal coupling after regressing the whole-brain global signal from both VTA and hippocampal time series. Coupling remained positive for all VTA–hippocampus pairs and was strongest for left VTA.
(B) Coupling control analyses averaged across all VTA–hippocampus pairs. Raw correlations were extremely high, consistent with strong shared global BOLD fluctuations. After global-signal regression, coupling was substantially reduced but remained robust. Comparable coupling was observed after first differencing of the global-residualized signals, indicating that residual VTA–hippocampal coordination was not explained solely by slow temporal drift.
(C) Sentence-boundary modulation of VTA–hippocampal coupling. Coupling within sentence-boundary windows was compared with coupling in matched random control windows. Positive Δr values indicate stronger coupling around sentence boundaries than around randomly selected time points. Boundary-related coupling enhancement was significant for all VTA–hippocampus pairs. Error bars indicate SEM across subjects. Stars denote one-sample tests against zero: * p < .05, ** p < .01, *** p < .001, **** p < 1e-4.
"""

    caption_file = OUT / "vta_hipp_coupling_residualized_boundary_controls_caption.txt"
    caption_file.write_text(caption)

    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {caption_file}")

    print("\nPanel B summary:")
    print(summary)

    print("\nPanel C summary:")
    print(bplot[[
        "vta_roi",
        "hipp_hemi",
        "mean_delta_z",
        "sem_delta_z",
        "mean_delta_r_approx",
        "t",
        "p",
    ]])


if __name__ == "__main__":
    main()