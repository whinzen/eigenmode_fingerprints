#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib

BASE = Path.home() / "eigenmode_fingerprints"
PANG = BASE / "pang_out"

INDEX = PANG / "subcortex" / "hippocampus_energy" / "hippocampus_energy_index.csv"
OUT = PANG / "subcortex" / "hippocampus_reconstruction"
OUT.mkdir(parents=True, exist_ok=True)

LABELS = {"L": 17, "R": 53}


def related_file(energy_file, suffix):
    return Path(str(energy_file).replace("_hipp_energy.npy", suffix))


def main():
    idx = pd.read_csv(INDEX)
    rows = []

    for _, r in idx.iterrows():
        sub = r["sub"]
        run = r["run"]
        hemi = r["hemi"]

        energy_file = Path(r["energy_file"])
        amp_file = related_file(energy_file, "_hipp_amplitudes.npy")
        phi_file = related_file(energy_file, "_hipp_phi.npy")
        bold_file = Path(r["bold_file"])
        seg_file = Path(r["seg_file"])

        if not amp_file.exists() or not phi_file.exists():
            print(f"[skip] missing amp/phi: {sub} {run} {hemi}")
            continue

        A = np.load(amp_file)      # K x T
        Phi = np.load(phi_file)    # V x K

        bold = nib.load(str(bold_file)).get_fdata()
        seg = nib.load(str(seg_file)).get_fdata()

        mask = seg == LABELS[hemi]

        X = bold[mask, :].astype(float)
        X = X - np.nanmean(X, axis=1, keepdims=True)

        T = min(X.shape[1], A.shape[1])
        X = X[:, :T]
        A = A[:, :T]

        total_ss = np.nansum(X ** 2)

        if total_ss <= 0 or not np.isfinite(total_ss):
            print(f"[skip] bad total variance: {sub} {run} {hemi}")
            continue

        max_k = min(A.shape[0], Phi.shape[1])

        for n_modes in range(1, max_k + 1):
            Phi_k = Phi[:, :n_modes]
            A_k = A[:n_modes, :]

            X_hat = Phi_k @ A_k

            resid_ss = np.nansum((X - X_hat) ** 2)
            r2 = 1.0 - resid_ss / total_ss

            rows.append({
                "sub": sub,
                "run": run,
                "hemi": hemi,
                "n_voxels": X.shape[0],
                "n_trs": T,
                "n_modes": n_modes,
                "r2_reconstruction": r2,
                "resid_ss": resid_ss,
                "total_ss": total_ss,
                "amp_file": str(amp_file),
                "phi_file": str(phi_file),
                "bold_file": str(bold_file),
            })

        print(
            f"✅ {sub} {run} {hemi}: "
            f"R2 1={rows[-max_k]['r2_reconstruction']:.3f}, "
            f"10={rows[-max_k+9]['r2_reconstruction']:.3f}, "
            f"30={rows[-1]['r2_reconstruction']:.3f}"
        )

    df = pd.DataFrame(rows)

    out_csv = OUT / "hippocampus_reconstruction_r2_by_run.csv"
    df.to_csv(out_csv, index=False)

    group = (
        df.groupby("n_modes", as_index=False)
        .agg(
            mean_r2=("r2_reconstruction", "mean"),
            sem_r2=("r2_reconstruction", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            median_r2=("r2_reconstruction", "median"),
            n=("r2_reconstruction", "size"),
        )
    )

    group_csv = OUT / "hippocampus_reconstruction_r2_group.csv"
    group.to_csv(group_csv, index=False)

    print(f"\nWrote {out_csv}")
    print(f"Wrote {group_csv}")
    print(group.head(30))


if __name__ == "__main__":
    main()