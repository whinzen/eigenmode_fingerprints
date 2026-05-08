import os
import sys
import numpy as np
import nibabel as nib

# ========== SETTINGS ==========
TR = 2.0
hemi_labels = ['L', 'R']
eigen_dir = "../eigenmodes"
func_dir = "../ds003643_derivs/derivatives/sub-EN057/func"
out_base = "../pang_out/energy"
os.makedirs(out_base, exist_ok=True)

# ========== PARSE RUN ID ==========
if len(sys.argv) < 2:
    print("Usage: python energy_per_run_timeseries.py --run_id <RUN_ID>")
    sys.exit(1)

try:
    run_id = int(sys.argv[sys.argv.index("--run_id") + 1])
except (ValueError, IndexError):
    print("❌ Invalid run_id input.")
    sys.exit(1)

print(f"Running energy extraction for run {run_id}")

# ========== LOOP OVER HEMISPHERES ==========
for hemi in hemi_labels:
    hemi_lower = hemi.lower()

    # Construct file paths
    func_path = os.path.join(func_dir, f"sub-EN057_task-lppEN_run-{run_id}_space-fsaverage5_desc-preproc_bold.{hemi_lower}.func.gii")
    eig_path = os.path.join(eigen_dir, f"fsaverage5.Laplacian_{hemi}.txt")
    out_dir = os.path.join(out_base, hemi)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"run-{run_id}.npy")

    print(f"Loading BOLD file: {func_path}")
    print(f"Loading eigenmodes: {eig_path}")

    try:
        img = nib.load(func_path)
        data = np.stack([d.data for d in img.darrays], axis=-1)
    except Exception as e:
        print(f"❌ Failed to load or extract GIfTI data: {e}")
        continue

    try:
        eigvecs = np.loadtxt(eig_path)
        if eigvecs.shape[0] != data.shape[0]:
            raise ValueError(f"Mismatch in vertices: {eigvecs.shape[0]} eigenmodes vs {data.shape[0]} vertices")
    except Exception as e:
        print(f"❌ Failed to load eigenmodes: {e}")
        continue

    # Project BOLD onto eigenmodes (time x modes)
    try:
        energy = np.dot(eigvecs.T, data)  # shape (K x T)
        energy = energy.T  # shape (T x K)
        np.save(out_path, energy)
        print(f"✅ Saved energy time series to: {out_path}")
    except Exception as e:
        print(f"❌ Failed to compute/save energy: {e}")