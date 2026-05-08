python - <<'PY'
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import csv

base = Path.home()/"eigenmode_fingerprints"
pang = base/"pang_out"
group = pang/"group"
group.mkdir(exist_ok=True)

E_all, subs = [], []
lam_ref = None

for summ in sorted(pang.glob("sub-*/summary")):
    p_lam = summ/"lam.npy"
    p_E   = summ/"avg_energy.npy"
    if not (p_lam.exists() and p_E.exists()):
        print(f"[warn] missing lam/E in {summ}"); continue
    lam = np.load(p_lam); E = np.load(p_E)
    if lam_ref is None: lam_ref = lam
    E_all.append(E); subs.append(summ.parent.name)

if not E_all:
    print("❌ No subject summaries found under", pang); raise SystemExit(1)

E = np.vstack(E_all)                 # (n_subj, K)
Emean = E.mean(axis=0)
Esem  = E.std(axis=0)/np.sqrt(E.shape[0])

np.save(group/"lam_group.npy", lam_ref)
np.save(group/"Emean_group.npy", Emean)
np.save(group/"Esem_group.npy", Esem)

# CSV
csv_path = group/"energy_spectrum_group.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["mode_index","lambda","Emean","Esem"])
    for k,(l,em,es) in enumerate(zip(lam_ref, Emean, Esem)):
        w.writerow([k, l, em, es])

# Plots
plt.figure()
plt.plot(np.arange(lam_ref.size), Emean)
plt.fill_between(np.arange(lam_ref.size), Emean-Esem, Emean+Esem, alpha=0.3)
plt.xlabel("Mode index k"); plt.ylabel("Group mean energy")
plt.title("Group: mean energy vs index")
plt.tight_layout(); plt.savefig(group/"group_energy_vs_index.png"); plt.close()

plt.figure()
plt.loglog(lam_ref, Emean, marker='o', linestyle='-')
plt.xlabel("Eigenvalue λ (log)"); plt.ylabel("Group mean energy (log)")
plt.title("Group: mean energy vs λ")
plt.tight_layout(); plt.savefig(group/"group_energy_vs_lambda.png"); plt.close()

print("✅ Saved group arrays + plots to", group)
print("Subjects aggregated:", len(subs), "→", subs)
print("CSV:", csv_path)
PY