# === rebuild_glm_np_per_run.py ===

import numpy as np
import nibabel as nib
from pathlib import Path
from glm import run_ols_and_save

BASE = Path.home() / "eigenmode_fingerprints"
FUNC_DIR = BASE / "data/empirical"
REGRESSOR_DIR = BASE / "pang_out/regressors/boundary_np_hrf_per_subject"
OUT_DIR = BASE / "pang_out/glm_per_run/boundary_np"
OUT_DIR.mkdir(parents=True, exist_ok=True)

subjects = sorted([p.name for p in FUNC_DIR.glob("sub-*")])
print(f"🔍 Found {len(subjects)} subjects.")

for subj in subjects:
    func_subj_dir = FUNC_DIR / subj / "func"
    if not func_subj_dir.exists():
        continue

    for func_file in sorted(func_subj_dir.glob("*_bold.func.gii")):
        fname = func_file.name

        try:
            run_id = int(fname.split("_run-")[1].split("_")[0])
        except Exception:
            print(f"⚠️ Could not parse run ID from {fname}")
            continue

        reg_path = REGRESSOR_DIR / f"{subj}_run-{run_id}.npy"
        if not reg_path.exists():
            print(f"⚠️ Missing regressor for {subj}_run-{run_id}.npy — skipping {fname}")
            continue

        X = np.load(reg_path)

        try:
            T = len(nib.load(func_file).darrays)
        except Exception as e:
            print(f"❌ Could not load {fname}: {e}")
            continue

        if X.shape[0] > T:
            X = X[:T]
        elif X.shape[0] < T:
            X = np.pad(X, (0, T - X.shape[0]))

        out_name = fname.replace("_bold.func.gii", "_glm_boundary_np.npy")
        out_path = OUT_DIR / subj / out_name
        out_path.parent.mkdir(exist_ok=True, parents=True)

        try:
            run_ols_and_save(
                func_file,
                X,
                out_path,
                run_id=run_id,
                predictor_names=["boundary_np"]
            )
            print(f"✅ GLM written: {out_path}")
        except Exception as e:
            print(f"❌ Error in {subj} run-{run_id}: {e}")

print("✅ Done with NP GLMs")