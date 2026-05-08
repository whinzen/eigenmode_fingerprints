import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Path to energy_spectrum_subject.csv (relative to code directory)
df = pd.read_csv("../pang_out/group/energy_spectrum_subject.csv")

# Define fitting range (excluding mode 0)
kmin, kmax = 1, 60
subjects = df["subject"].unique()

plt.figure(figsize=(7, 5))

# Plot subject-level fits in log–log λ vs E_k space
for subj in subjects:
    sub_df = df[(df["subject"] == subj) & (df["mode_k"].between(kmin, kmax))]
    x = sub_df["lam"].values
    y = sub_df["Emean"].values
    if len(x) < 2 or np.any(y <= 0):
        continue
    logx = np.log10(x)
    logy = np.log10(y)
    coeffs = np.polyfit(logx, logy, 1)
    y_fit = 10**np.poly1d(coeffs)(logx)
    plt.loglog(x, y_fit, color="gray", alpha=0.3, lw=1)

# Group mean spectrum over subjects in fitting range
group = df[df["mode_k"].between(kmin, kmax)].groupby("mode_k", as_index=False).agg({
    "lam": "first",
    "Emean": "mean"
})

# Fit group mean in log–log space
x_fit = group["lam"].values
y_fit = group["Emean"].values
logx = np.log10(x_fit)
logy = np.log10(y_fit)
group_coeffs = np.polyfit(logx, logy, 1)
slope, intercept = group_coeffs
y_fit_line = 10**np.poly1d(group_coeffs)(logx)

# Plot group mean and fit
plt.loglog(x_fit, y_fit, lw=2.5, color="blue", label=f"Group Mean (k={kmin}-{kmax})")
plt.loglog(x_fit, y_fit_line, "--", color="black", lw=2,
           label=f"Fit (k={kmin}-{kmax})\nSlope = {slope:.2f}")

plt.xlabel("Eigenvalue λ (log)")
plt.ylabel("Energy $E_k$ (log)")
plt.title("Group + Subject Criticality Fit (log–log λ–E_k, zoomed)")
plt.legend()
plt.tight_layout()

# Save plot relative to current (code) directory
plt.savefig("../pang_out/group/group_energy_fit_loglog_subjects_zoom.png", dpi=150)
plt.show()