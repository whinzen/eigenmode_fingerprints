(pang_vtk) python - <<'PY'
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

base      = Path.home()/"eigenmode_fingerprints"
pang_root = base/"pang_out"
modes_dir = base/"data/template_eigenmodes/fsaverage5"

# eigenvalues (shared across subjects/runs on fsaverage5)
lam = np.load(modes_dir/"lam_L.npy")  # same as lam_R

made_any = False
for sub_dir in sorted(pang_root.glob("sub-*")):
    run_dirs = sorted(sub_dir.glob("run-*"))
    if not run_dirs:
        print(f"[skip] no runs found under {sub_dir.name}")
        continue

    Eruns = []
    for rd in run_dirs:
        pEL, pER = rd/"E_L.npy", rd/"E_R.npy"
        if not (pEL.exists() and pER.exists()):
            print(f"[warn] missing E files in {rd}")
            continue
        EL, ER = np.load(pEL), np.load(pER)     # shape (K,)
        Eruns.append(0.5*(EL+ER))
    if not Eruns:
        print(f"[skip] no usable runs for {sub_dir.name}")
        continue

    E = np.vstack(Eruns)              # (n_runs, K)
    Emean = E.mean(axis=0)            # (K,)
    Esem  = E.std(axis=0)/np.sqrt(E.shape[0])

    summ = sub_dir/"summary"
    summ.mkdir(exist_ok=True)
    np.save(summ/"lam.npy", lam)
    np.save(summ/"avg_energy.npy", Emean)
    np.save(summ/"sem_energy.npy", Esem)

    # quick plots
    plt.figure()
    plt.plot(np.arange(lam.size), Emean)
    plt.fill_between(np.arange(lam.size), Emean-Esem, Emean+Esem, alpha=0.3)
    plt.xlabel("Mode index k"); plt.ylabel("Mean energy")
    plt.title(f"{sub_dir.name}: avg energy vs index")
    plt.tight_layout(); plt.savefig(summ/"avg_energy_vs_index.png"); plt.close()

    plt.figure()
    plt.loglog(lam, Emean, marker='o', linestyle='-')
    plt.xlabel("Eigenvalue λ (log)"); plt.ylabel("Mean energy (log)")
    plt.title(f"{sub_dir.name}: avg energy vs λ")
    plt.tight_layout(); plt.savefig(summ/"avg_energy_vs_lambda.png"); plt.close()

    made_any = True
    print(f"✅ {sub_dir.name}: summary saved to {summ}  (runs used: {E.shape[0]})")

if not made_any:
    print("❌ No subject summaries were created. Check pang_out/sub-*/run-* for E_L.npy / E_R.npy.")
PY