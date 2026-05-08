python - <<'PY'
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def aicc(n, rss, k_params):
    # AICc for Gaussian residuals with constant variance
    # AIC = n*log(RSS/n) + 2k;  AICc = AIC + (2k(k+1))/(n-k-1)
    AIC = n*np.log(rss/n) + 2*k_params
    return AIC + (2*k_params*(k_params+1))/(n - k_params - 1) if n > k_params+1 else np.inf

base = Path.home()/ "eigenmode_fingerprints/pang_out/group"
lam  = np.load(base/"lam_group.npy")
E    = np.load(base/"Emean_group.npy")
Esem = np.load(base/"Esem_group.npy") if (base/"Esem_group.npy").exists() else None

# Fit range: nonzero modes k=1..60
kmax = min(60, len(lam)-1)
idx  = np.arange(1, kmax+1)
x_ll = np.log10(lam[idx])     # for power-law
y_ll = np.log10(E[idx])
x_exp= lam[idx]               # for exponential
y_ln = np.log(E[idx])

# --- Model 1: single power law (OLS) ---
A = np.c_[np.ones_like(x_ll), x_ll]
coef_pl, _, _, _ = np.linalg.lstsq(A, y_ll, rcond=None)
yhat_pl = A @ coef_pl
rss_pl = np.sum((y_ll - yhat_pl)**2)
aicc_pl = aicc(len(idx), rss_pl, k_params=2)
slope_pl = coef_pl[1]

# --- Model 2: single exponential (OLS on log E vs λ) ---
A2 = np.c_[np.ones_like(x_exp), x_exp]
coef_exp, _, _, _ = np.linalg.lstsq(A2, y_ln, rcond=None)
yhat_exp = A2 @ coef_exp
rss_exp = np.sum((y_ln - yhat_exp)**2)
aicc_exp = aicc(len(idx), rss_exp, k_params=2)
slope_exp = coef_exp[1]

# --- Model 3: broken power law (one unknown breakpoint, brute force over candidate cuts) ---
best = {"aicc": np.inf}
for cut in range(3, len(idx)-3):  # leave ≥3 points each side
    left  = slice(0, cut)
    right = slice(cut, len(idx))
    # left fit
    A_L = np.c_[np.ones_like(x_ll[left]),  x_ll[left]]
    cL, *_ = np.linalg.lstsq(A_L, y_ll[left], rcond=None)
    yL = A_L @ cL
    rssL = np.sum((y_ll[left] - yL)**2)
    # right fit
    A_R = np.c_[np.ones_like(x_ll[right]), x_ll[right]]
    cR, *_ = np.linalg.lstsq(A_R, y_ll[right], rcond=None)
    yR = A_R @ cR
    rssR = np.sum((y_ll[right] - yR)**2)
    rss  = rssL + rssR
    # params: 4 (a,b for each segment) + 1 breakpoint index (treated as an extra parameter)
    aicc_bpl = aicc(len(idx), rss, k_params=5)
    if aicc_bpl < best["aicc"]:
        best = {"aicc": aicc_bpl, "cut": cut, "cL": cL, "cR": cR}

print(f"Fit on k=1..{kmax} (N={len(idx)})")
print(f" Power-law:   slope b={slope_pl:.3f},  AICc={aicc_pl:.1f}")
print(f" Exponential: slope b={slope_exp:.3e}, AICc={aicc_exp:.1f} (on logE vs λ)")
print(f" Broken PL:   cut@k={idx[best['cut']]}, AICc={best['aicc']:.1f}")

# Which wins?
aiccs = {"PowerLaw": aicc_pl, "Exponential": aicc_exp, "BrokenPL": best["aicc"]}
order = sorted(aiccs.items(), key=lambda t: t[1])
print(" Model ranking (lower is better):")
for name, val in order:
    print(f"  - {name}: AICc={val:.1f}")
print(f" ΔAICc vs best:")
best_val = order[0][1]
for name, val in aiccs.items():
    print(f"  - {name}: {val-best_val:+.1f}")

# Plot (with optional SEM shading) in log-log for visual check
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 140
fig, ax = plt.subplots(figsize=(4.2,3.4))
ax.plot(np.log10(lam[idx]), np.log10(E[idx]), '.', ms=4, label='mean')

if Esem is not None:
    # error band for log10(E): use delta(log10 y) ≈ (Esem / (E ln 10))
    import numpy as np
    y = np.log10(E[idx])
    dy = (Esem[idx] / (E[idx]*np.log(10))).clip(max=0.5)  # cap to avoid huge bands at tiny E
    ax.fill_between(np.log10(lam[idx]), y-dy, y+dy, alpha=0.2, label='SEM')

# overlay the single power-law fit
ax.plot(np.log10(lam[idx]), yhat_pl, '-', lw=2, label=f"Power-law fit (b={slope_pl:.2f})")

# and the broken fit
cut = best["cut"]
A_L = np.c_[np.ones_like(x_ll[:cut]),  x_ll[:cut]]
A_R = np.c_[np.ones_like(x_ll[cut:]), x_ll[cut:]]
yL = A_L @ best["cL"]; yR = A_R @ best["cR"]
ax.plot(np.log10(lam[idx[:cut]]), yL, '--', lw=1.8, label=f"Broken PL (left)")
ax.plot(np.log10(lam[idx[cut:]]), yR, '--', lw=1.8, label=f"Broken PL (right)")

ax.set_xlabel(r'$\log_{10}\lambda$')
ax.set_ylabel(r'$\log_{10} \mathrm{Energy}$')
ax.legend(frameon=False, fontsize=8)
ax.set_title(f'Group energy spectrum (k=1..{kmax})')
fig.tight_layout()
outp = base/"group_model_compare_loglog.png"
fig.savefig(outp, dpi=180)
plt.close(fig)
print("Saved figure:", outp)
PY