from pathlib import Path

# Define paths
BASE = Path.home() / "eigenmode_fingerprints"
FUNC_DIR = BASE / "data/empirical"
REGRESSOR_DIR = BASE / "pang_out/regressors/boundary_np_hrf_per_subject"

# Start
print("🔍 Checking regressor coverage...")

subjects = sorted([p.name for p in FUNC_DIR.glob("sub-*")])
n_total = 0
n_found = 0

for subj in subjects:
    func_dir = FUNC_DIR / subj / "func"
    bold_files = sorted(func_dir.glob("*_run-*_bold.func.gii"))

    for f in bold_files:
        fname = f.name
        run_str = fname.split("_run-")[1].split("_")[0]
        run_id = int(run_str)
        reg_file = REGRESSOR_DIR / f"{subj}_run-{run_id}.npy"
        n_total += 1

        if reg_file.exists():
            print(f"✅ Found:  {reg_file.name}")
            n_found += 1
        else:
            print(f"❌ Missing: {reg_file.name}")

print("\n📊 Summary:")
print(f"  Total expected regressors: {n_total}")
print(f"  Found: {n_found}")
print(f"  Missing: {n_total - n_found}")