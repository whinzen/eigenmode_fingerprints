import numpy as np
import nibabel as nib
from pathlib import Path

# === config ===
base = Path.home() / "eigenmode_fingerprints"
empirical = base / "data" / "empirical"                # contains sub-*/ subfolders
modes_dir = base / "data" / "template_eigenmodes" / "fsaverage5"
out_root  = base / "pang_out"

# load template eigenmodes once
phi_L = np.load(modes_dir / "phi_L.npy")
phi_R = np.load(modes_dir / "phi_R.npy")
lam_L = np.load(modes_dir / "lam_L.npy")
lam_R = np.load(modes_dir / "lam_R.npy")

def load_func_gii(path: Path) -> np.ndarray:
    g = nib.load(str(path))
    return np.column_stack([da.data for da in g.darrays]).astype(float)

def standardize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd

def process_run(sub: str, run: str, funcL: Path, funcR: Path):
    XL = standardize(load_func_gii(funcL))
    XR = standardize(load_func_gii(funcR))

    AL = phi_L.T @ XL
    AR = phi_R.T @ XR
    EL = (AL**2).mean(axis=1)
    ER = (AR**2).mean(axis=1)

    out_dir = out_root / sub / run
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "A_L.npy", AL)
    np.save(out_dir / "A_R.npy", AR)
    np.save(out_dir / "E_L.npy", EL)
    np.save(out_dir / "E_R.npy", ER)
    np.save(out_dir / "lam_L.npy", lam_L)
    np.save(out_dir / "lam_R.npy", lam_R)

    print(f"[{sub}-{run}] OK | T={AL.shape[1]} | out={out_dir}")

def main():
    # iterate subjects (sub-*/ folders inside empirical/)
    for sub_dir in sorted(empirical.glob("sub-*")):
        sub = sub_dir.name
        # find all LEFT hemi runs for this subject (recursively)
        pattern = f"{sub}_task-*_run-*_hemi-L_space-fsaverage5_bold.func.gii"
        for funcL in sorted(sub_dir.rglob(pattern)):
            # pair with RIGHT
            funcR = funcL.with_name(funcL.name.replace("hemi-L", "hemi-R"))
            if not funcR.exists():
                print(f"[warn] missing R: {funcR}")
                continue
            # parse run from name (e.g., run-15)
            parts = funcL.name.split("_")
            run = next((p for p in parts if p.startswith("run-")), "run-UNK")
            process_run(sub, run, funcL, funcR)

if __name__ == "__main__":
    main()