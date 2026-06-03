#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

IN = PANG / "subcortex" / "hippocampus_reconstruction" / "hippocampus_reconstruction_r2_group.csv"
OUT = PANG / "subcortex" / "hippocampus_reconstruction"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(IN)

    fig, ax = plt.subplots(figsize=(6.2, 4.5))

    x = df["n_modes"].values
    y = df["mean_r2"].values
    sem = df["sem_r2"].values

    ax.plot(x, y, marker="o", linewidth=2, markersize=4)
    ax.fill_between(x, y - sem, y + sem, alpha=0.2, linewidth=0)

    ax.axhline(0.90, linestyle="--", linewidth=1)
    ax.axhline(0.80, linestyle="--", linewidth=1)

    ax.set_xlabel("Number of hippocampal eigenmodes")
    ax.set_ylabel("Reconstruction $R^2$")
    ax.set_title("Cumulative BOLD variance captured by hippocampal eigenmodes")
    ax.set_ylim(0.70, 0.93)
    ax.set_xlim(1, 30)

    ax.text(30, 0.90, "90%", va="bottom", ha="right", fontsize=9)
    ax.text(30, 0.80, "80%", va="bottom", ha="right", fontsize=9)

    fig.tight_layout()

    out_png = OUT / "hippocampus_reconstruction_r2.png"
    out_pdf = OUT / "hippocampus_reconstruction_r2.pdf"

    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)

    caption = """Figure X. Hippocampal eigenmode reconstruction performance. Cumulative reconstruction R² is shown as a function of the number of retained hippocampal graph eigenmodes. For each subject, run, and hemisphere, voxelwise hippocampal BOLD signals were mean-centered and reconstructed from the first K eigenmodes using the saved hippocampal eigenvectors and mode amplitudes. Reconstruction performance was computed as 1 minus residual sum of squares divided by total sum of squares. The first hippocampal eigenmode alone captured approximately 76% of voxelwise BOLD variance, and the first 30 modes captured approximately 90%, indicating that hippocampal activity is highly compressible within the retained graph-eigenmode basis. Shaded area indicates SEM across subject-run-hemisphere observations."""
    (OUT / "hippocampus_reconstruction_r2_caption.txt").write_text(caption)

    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {OUT / 'hippocampus_reconstruction_r2_caption.txt'}")


if __name__ == "__main__":
    main()