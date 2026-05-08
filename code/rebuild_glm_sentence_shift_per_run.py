import numpy as np
from pathlib import Path
from scipy.stats import zscore

BASE = Path.home() / "eigenmode_fingerprints"
DATA = BASE / "pang_out"
REG = DATA / "regressors/sentence_shift_hrf_per_subject"

SUBS = sorted([p for p in DATA.glob("sub-*") if p.is_dir()])

for sub in SUBS:
    out_dir = sub / "glm_sentence_shift"
    out_dir.mkdir(exist_ok=True)

    rows = []

    for f in REG.glob("run-*.npy"):
        run = int(f.stem.split("-")[1])
        x = np.load(f)

        for hemi in ["L", "R"]:
            A = np.load(sub / f"run-{run:02d}_hemi-{hemi}_A.npy")
            E = A ** 2

            if E.shape[1] != len(x):
                continue

            xz = zscore(x)
            for k in range(E.shape[0]):
                y = zscore(E[k])
                beta = np.dot(xz, y) / len(y)
                rows.append([run, hemi, k+1, beta])

    np.savetxt(out_dir / "betas.csv", rows, delimiter=",",
               header="run,hemi,mode_k,beta", comments="")