#!/usr/bin/env python

from pathlib import Path
import re
import numpy as np
import nibabel as nib
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigsh

BASE = Path.home() / "eigenmode_fingerprints"
DATA = BASE / "data" / "empirical"
OUT = BASE / "pang_out" / "subcortex" / "hippocampus_energy"
OUT.mkdir(parents=True, exist_ok=True)

N_MODES = 30
LABELS = {"L": 17, "R": 53}


def parse_run(path):
    m = re.search(r"(run-\d+)", path.name)
    return m.group(1) if m else "run-NA"


def build_adjacency(mask):
    coords = np.array(np.where(mask)).T
    idx = {tuple(c): i for i, c in enumerate(coords)}

    neigh = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1],
    ])

    rows, cols = [], []
    for i, c in enumerate(coords):
        for d in neigh:
            j = idx.get(tuple(c + d))
            if j is not None:
                rows.append(i)
                cols.append(j)

    n = len(coords)
    A = sparse.coo_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(n, n)
    ).tocsr()
    return A, coords


def compute_modes(mask):
    A, coords = build_adjacency(mask)
    D = sparse.diags(np.asarray(A.sum(axis=1)).ravel())
    L = D - A

    n = L.shape[0]
    k = min(N_MODES, n - 2)

    vals, vecs = eigsh(L, k=k, which="SM")
    order = np.argsort(vals)

    return vals[order], vecs[:, order], coords


def project_bold(bold, mask, phi):
    X = bold[mask, :].astype(float)          # V x T
    X -= np.nanmean(X, axis=1, keepdims=True)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    A = phi.T @ X                            # K x T
    E = A ** 2                               # K x T
    return A, E


def process_run(bold_file):
    sub = bold_file.parts[-3]
    run = parse_run(bold_file)

    seg_file = bold_file.with_name(
        bold_file.name.replace(
            "_desc-preproc_bold.nii.gz",
            "_desc-aseg_dseg.nii.gz"
        )
    )

    if not seg_file.exists():
        print(f"[skip] missing aseg: {seg_file}")
        return []

    bold_img = nib.load(str(bold_file))
    seg_img = nib.load(str(seg_file))

    if bold_img.shape[:3] != seg_img.shape[:3]:
        print(f"[skip] shape mismatch: {bold_file}")
        return []

    bold = bold_img.get_fdata()
    seg = seg_img.get_fdata().astype(int)

    rows = []

    for hemi, label in LABELS.items():
        mask = seg == label
        n_vox = int(mask.sum())

        if n_vox < 20:
            print(f"[skip] too few voxels {sub} {run} hemi-{hemi}: {n_vox}")
            continue

        lam, phi, coords = compute_modes(mask)
        A, E = project_bold(bold, mask, phi)

        stem = f"{sub}_{run}_hemi-{hemi}_hipp"
        np.save(OUT / f"{stem}_lam.npy", lam)
        np.save(OUT / f"{stem}_phi.npy", phi)
        np.save(OUT / f"{stem}_coords.npy", coords)
        np.save(OUT / f"{stem}_amplitudes.npy", A)
        np.save(OUT / f"{stem}_energy.npy", E)

        rows.append({
            "sub": sub,
            "run": run,
            "hemi": hemi,
            "n_voxels": n_vox,
            "n_modes": E.shape[0],
            "n_trs": E.shape[1],
            "bold_file": str(bold_file),
            "seg_file": str(seg_file),
            "energy_file": str(OUT / f"{stem}_energy.npy"),
        })

        print(f"✅ {sub} {run} hemi-{hemi}: vox={n_vox}, E={E.shape}")

    return rows


def main():
    bold_files = sorted(DATA.glob(
        "sub-EN*/func/*task-lppEN*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    ))

    print("Found BOLD files:", len(bold_files))

    all_rows = []
    for f in bold_files:
        all_rows.extend(process_run(f))

    index = pd.DataFrame(all_rows)
    index_file = OUT / "hippocampus_energy_index.csv"
    index.to_csv(index_file, index=False)

    print("\n✅ wrote", index_file)
    print(index.groupby(["hemi"])["energy_file"].count())


if __name__ == "__main__":
    main()