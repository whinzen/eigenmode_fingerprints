#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

INFILE = (
    PANG /
    "subcortex" /
    "hipp_mean_signal_AP_group" /
    "group_all_predictors_AP_by_hemi.csv"
)

OUTDIR = (
    PANG /
    "subcortex" /
    "hipp_mean_signal_AP_group"
)

ORDER = [
    "sentence_boundary",
    "sentence_shift",
    "token_shift",
    "pred_error_ar",
    "pred_error_subspace",
    "curvature",
]

PARCELS = [
    "anterior",
    "middle",
    "posterior",
]


def main():

    df = pd.read_csv(INFILE)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        sharey=True
    )

    for ax, hemi in zip(
        axes,
        ["L", "R"]
    ):

        d = df[df["hemi"] == hemi]

        x = np.arange(len(ORDER))

        width = 0.25

        for i, parcel in enumerate(PARCELS):

            p = d[d["parcel"] == parcel]

            p["predictor"] = pd.Categorical(
                p["predictor"],
                ORDER,
                ordered=True
            )

            p = p.sort_values("predictor")

            ax.bar(
                x + (i - 1) * width,
                p["beta_mean"],
                width=width,
                yerr=p["beta_sem"],
                capsize=3,
                label=parcel,
            )

        ax.axhline(0, lw=1)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [s.replace("_", "\n") for s in ORDER],
            fontsize=8,
        )

        ax.set_title(
            f"{hemi} hippocampus"
        )

    axes[0].set_ylabel("β")

    axes[1].legend(
        frameon=False
    )

    plt.tight_layout()

    out_png = (
        OUTDIR /
        "hipp_AP_profiles.png"
    )

    out_pdf = (
        OUTDIR /
        "hipp_AP_profiles.pdf"
    )

    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf)

    print(out_png)
    print(out_pdf)


if __name__ == "__main__":
    main()