#!/usr/bin/env python

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

BASE = Path.home() / "eigenmode_fingerprints"
IN_CSV = BASE / "pang_out" / "mode3_annotations" / "mode3_neuromaps_correlations.csv"
OUT_DIR = BASE / "pang_out" / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN_CSV)
d = df[df["hemi"] == "bihemi"].copy()
d = d.sort_values("abs_pearson_r", ascending=True)

fig, ax = plt.subplots(figsize=(6.5, 3.5))

ax.barh(d["annotation"], d["abs_pearson_r"])
ax.set_xlabel("|Pearson r| with Mode 3")
ax.set_ylabel("")
ax.set_title("Mode 3 alignment with canonical cortical maps")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

png = OUT_DIR / "figure_mode3_neuromaps_annotation.png"
pdf = OUT_DIR / "figure_mode3_neuromaps_annotation.pdf"

plt.savefig(png, dpi=300, facecolor="white")
plt.savefig(pdf, facecolor="white")
plt.close()

print(f"✅ wrote {png}")
print(f"✅ wrote {pdf}")