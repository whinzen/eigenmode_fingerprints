import numpy as np
from pathlib import Path

base = Path.home()/ "eigenmode_fingerprints"/"pang_out"/"group"

lam   = np.load(base/"group_lambda.npy")
Emean = np.load(base/"group_energy_mean.npy")
Esem  = np.load(base/"group_energy_sem.npy")

print("λ (first 10):", lam[:10])
print("Emean (first 10):", Emean[:10])
print("Esem (first 10):", Esem[:10])

print("\nSummary:")
print("λ range:", lam.min(), "→", lam.max())
print("Mean energy range:", Emean.min(), "→", Emean.max())
