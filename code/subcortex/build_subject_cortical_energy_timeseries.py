#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib

BASE = Path.home() / "eigenmode_fingerprints"
EMP = BASE / "data" / "empirical"
EIG = BASE / "modes" / "fsaverage5"
OUT = BASE / "pang_out" / "cortical_energy_subject"

PHI = {
    "L": np.load(EIG / "phi_L.npy"),
    "R": np.load(EIG / "phi_R.npy"),
}

OUT.mkdir(parents=True, exist_ok=True)


def load_gii(path):
    gii = nib.load(str(path))
    d = np.stack([arr.data for arr in gii.darrays])  # [T, V]
    return d.T  # [V, T]


def zscore_vertices(X):
    mu = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, keepdims=True)
    sd[~np.isfinite(sd)] = 1.0
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def parse_run(fname):
    return fname.split("_run-")[1].split("_")[0]


def main():
    rows = []

    for sub_dir in sorted(EMP.glob("sub-*")):
        func = sub_dir / "func"
        if not func.exists():
            continue

        subject = sub_dir.name

        for hemi in ["L", "R"]:
            pattern = f"*run-*_hemi-{hemi}_space-fsaverage5_bold.func.gii"
            files = sorted(func.glob(pattern))

            for f in files:
                run = parse_run(f.name)

                try:
                    X = load_gii(f)              # [V,T]
                    Xz = zscore_vertices(X)
                    A = PHI[hemi].T @ Xz        # [K,T]
                    E = A ** 2                  # [K,T]

                    out_dir = OUT / subject
                    out_dir.mkdir(parents=True, exist_ok=True)

                    out_file = out_dir / f"{subject}_run-{int(run):02d}_hemi-{hemi}_cortical_energy.npy"
                    np.save(out_file, E)

                    rows.append({
                        "subject": subject,
                        "run": f"{int(run):02d}",
                        "hemi": hemi,
                        "n_modes": E.shape[0],
                        "n_trs": E.shape[1],
                        "bold_file": str(f),
                        "energy_file": str(out_file),
                    })

                    print(f"✅ {subject} run-{int(run):02d} hemi-{hemi} {E.shape}")

                except Exception as e:
                    print(f"[WARN] failed {subject} run-{run} hemi-{hemi}: {e}")

    idx = pd.DataFrame(rows)
    idx_file = OUT / "cortical_energy_subject_index.csv"
    idx.to_csv(idx_file, index=False)

    print(f"\nWrote {idx_file}")
    print(idx.head())


if __name__ == "__main__":
    main()