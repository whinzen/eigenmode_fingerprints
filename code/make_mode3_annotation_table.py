#!/usr/bin/env python

import pandas as pd
from pathlib import Path

BASE = Path.home() / "eigenmode_fingerprints"
IN_CSV = BASE / "pang_out" / "mode3_annotations" / "mode3_neuromaps_correlations.csv"
OUT_DIR = BASE / "pang_out" / "paper_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = {
    "Margulies_FC_gradient1": "Margulies gradient 1",
    "HCP_myelin_T1wT2w": "Myelin (T1w/T2w)",
    "HCP_cortical_thickness": "Cortical thickness",
}

ORDER = [
    "Margulies_FC_gradient1",
    "HCP_myelin_T1wT2w",
    "HCP_cortical_thickness",
]


def main():
    df = pd.read_csv(IN_CSV)

    # Use hemisphere-specific values only.
    # Bihemispheric correlations can cancel because eigenmode signs are arbitrary
    # and can flip across hemispheres.
    df = df[df["hemi"].isin(["L", "R"])].copy()

    rows = []

    for ann in ORDER:
        d = df[df["annotation"] == ann].copy()
        if d.empty:
            continue

        l = d[d["hemi"] == "L"]
        r = d[d["hemi"] == "R"]

        row = {
            "Canonical map": LABELS.get(ann, ann),
            "|Pearson r| L": abs(float(l["pearson_r"].iloc[0])) if len(l) else None,
            "|Pearson r| R": abs(float(r["pearson_r"].iloc[0])) if len(r) else None,
            "|Spearman rho| L": abs(float(l["spearman_rho"].iloc[0])) if len(l) else None,
            "|Spearman rho| R": abs(float(r["spearman_rho"].iloc[0])) if len(r) else None,
        }

        rows.append(row)

    out = pd.DataFrame(rows)

    # Round for manuscript table
    numeric_cols = [c for c in out.columns if c != "Canonical map"]
    for c in numeric_cols:
        out[c] = out[c].round(2)

    out_csv = OUT_DIR / "table_mode3_canonical_map_correlations_clean.csv"
    out.to_csv(out_csv, index=False)

    print("✅ Clean Mode 3 annotation table")
    print(out.to_string(index=False))
    print(f"\n✅ wrote {out_csv}")


if __name__ == "__main__":
    main()