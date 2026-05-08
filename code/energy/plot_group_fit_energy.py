import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# CONFIG
BASE = Path.home() / "eigenmode_fingerprints"
FNAME = BASE / "pang_out" / "group" / "energy_spectrum_subject.csv"
OUTFIG = BASE / "pang_out" / "group" / "group_energy_fit_loglog_subjects.png"

kmin, kmax = 6, 60

# Load subject-level data
df = pd.read_csv(FNAME)
subjects = df["subject"].unique()

plt.figure(figsize=(7, 5))

# Plot individual subject log-log fits
for subj in subjects:
    sub_df = df[(df["subject"] == subj) &
                (df["mode_k"] >= kmin) & (df["mode_k"] <= kmax)]
    x = sub_df["lam"].values
    y = sub_df["Emean"].values
    if len(x) < 2 or np.any(y <= 0) or np.any(x <= 0):
        continue
    logx = np.log10(x)
    logy = np.log10(y)
    coeffs = np.polyfit(logx, logy, 1)
    y_fit = 10**np.poly1d(coeffs)(logx)
    plt.plot(x, y_fit, color="gray", alpha=0.25, lw=1)

# Group-level mean and fit
group = df.groupby("mode_k").agg({"lam": "first", "Emean": "mean"}).reset_index()
group_fit = group[(group["mode_k"] >= kmin) & (group["mode_k"] <= kmax)]
xg = group_fit["lam"].values
yg = group_fit["Emean"].values
logxg = np.log10(xg)
logyg = np.log10(yg)
coeffs_group = np.polyfit(logxg, logyg, 1)
slope, intercept = coeffs_group
y_fit_group = 10**np.poly1d(coeffs_group)(logxg)

# Plot group mean and fit
plt.loglog(group["lam"], group["Emean"], lw=2.5, color="blue", label="Group Mean")
plt.loglog(xg, y_fit_group, "--", color="black", lw=2,
           label=f"Fit (k={kmin}–{kmax})\nSlope = {slope:.2f}")
plt.xlabel("Eigenvalue λ (log)")
plt.ylabel("Energy $E_k$ (log)")
plt.title("Group + Subject Criticality Fit (log–log λ–E_k)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTFIG, dpi=150)
plt.show()

print("✅ Saved:", OUTFIG)