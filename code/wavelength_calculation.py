python - <<'PY'
import numpy as np, pandas as pd
from pathlib import Path
base = Path.home()/ "eigenmode_fingerprints/pang_out/group"
lam = np.load(base/"lam_group.npy")
Emean = np.load(base/"Emean_group.npy")
Esem  = np.load(base/"Esem_group.npy") if (base/"Esem_group.npy").exists() else None

# Exclude the DC mode at index 0
idx = np.arange(1, len(lam))
lam_nz = lam[idx]
wl_mm = (2*np.pi)/np.sqrt(lam_nz)

df = pd.DataFrame({
    "k": idx,                 # mode index (nonzero modes)
    "lambda": lam_nz,
    "wavelength_mm": wl_mm,
    "Emean": Emean[idx],
    **({"Esem": Esem[idx]} if Esem is not None else {})
})
out_csv = base/"wavelength_table.csv"
df.to_csv(out_csv, index=False)
print("Wrote:", out_csv)
print(df.head(10))
PY