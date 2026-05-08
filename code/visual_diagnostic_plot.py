python - <<'PY'
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import math

base   = Path.home() / "eigenmode_fingerprints"
gdir   = base / "pang_out" / "group"           # where your group files are
gdir.mkdir(parents=True, exist_ok=True)

# --- Load group arrays ---
# Expect these from your earlier group step:
lam   = np.load(gdir/"lam_group.npy")          # (K,)
Emean = np.load(gdir/"Emean_group.npy")        # (K,)
Esem  = np.load(gdir/"Esem_group.npy")         # (K,)
# Optional: precomputed good-k mask; fall back to first 60 if missing
idx_path = gdir/"idx_group.npy"
if idx_path.exists():
    idx = np.load(idx_path).astype(bool)
else:
    idx = np.zeros_like(lam, dtype=bool)
    idx[:min(60, len(idx))] = True

# ---- Helpers ----
def aicc_from_rss(n, k, rss):
    # AICc for Gaussian errors: n*ln(RSS/n) + 2k + 2k(k+1)/(n-k-1)
    if rss <= 0:  # guard
        rss = 1e-12
    aic = n * math.log(rss / n) + 2 * k
    if n - k - 1 > 0:
        aic += (2 * k * (k + 1)) / (n - k - 1)
    return aic

def fit_powerlaw(xl, yl, wl=None):
    # linear fit: log10(E) = a + b * log10(lambda)
    if wl is None:
        p = np.polyfit(xl, yl, 1)
        b, a = p[0], p[1]
        yhat = a + b*xl
        rss  = np.sum((yl - yhat)**2)
    else:
        # weighted least squares
        W = np.diag(wl)
        X = np.column_stack([np.ones_like(xl), xl])
        beta = np.linalg.inv(X.T@W@X) @ (X.T@W@yl)
        a, b = beta[0], beta[1]
        yhat = a + b*xl
        rss  = float((yl - yhat).T @ W @ (yl - yhat))
    aicc = aicc_from_rss(len(xl), 2, rss)
    return a, b, yhat, rss, aicc

def fit_exponential(x_lin, yl, wl=None):
    # linear in λ: log10(E) = a + b * λ
    if wl is None:
        p = np.polyfit(x_lin, yl, 1)
        b, a = p[0], p[1]
        yhat = a + b*x_lin
        rss  = np.sum((yl - yhat)**2)
    else:
        W = np.diag(wl)
        X = np.column_stack([np.ones_like(x_lin), x_lin])
        beta = np.linalg.inv(X.T@W@X) @ (X.T@W@yl)
        a, b = beta[0], beta[1]
        yhat = a + b*x_lin
        rss  = float((yl - yhat).T @ W @ (yl - yhat))
    aicc = aicc_from_rss(len(x_lin), 2, rss)
    return a, b, yhat, rss, aicc

def fit_broken_powerlaw(xl, yl, kmin=2, kmax=12, wl=None):
    # try cut points in [kmin..kmax) over the selected domain
    best = None
    n = len(xl)
    for cut in range(kmin, min(kmax, n-1)):
        xl1, yl1 = xl[:cut], yl[:cut]
        xl2, yl2 = xl[cut:], yl[cut:]
        if wl is None:
            a1,b1,y1,rss1,aic1 = fit_powerlaw(xl1, yl1)
            a2,b2,y2,rss2,aic2 = fit_powerlaw(xl2, yl2)
        else:
            w1, w2 = wl[:cut], wl[cut:]
            a1,b1,y1,rss1,aic1 = fit_powerlaw(xl1, yl1, wl=w1)
            a2,b2,y2,rss2,aic2 = fit_powerlaw(xl2, yl2, wl=w2)
        # Combined AICc with k=4 parameters (two a,b pairs)
        rss = rss1 + rss2
        aicc = aicc_from_rss(n, 4, rss)
        if (best is None) or (aicc < best["AICc"]):
            best = dict(cut=cut, a1=a1, b1=b1, a2=a2, b2=b2,
                        yhat=np.concatenate([y1,y2]), RSS=rss, AICc=aicc)
    return best

# ---- Prepare data (selected band) ----
sel = np.where(idx)[0]
xl   = np.log10(lam[sel])
yl   = np.log10(Emean[sel])
# Weight by inverse variance on log scale (approx): w ~ 1/(SEM/E)^2 = (E/SEM)^2
sem_sel = Esem[sel]
with np.errstate(divide='ignore', invalid='ignore'):
    w = (Emean[sel] / np.where(sem_sel>0, sem_sel, np.inf))**2
    w = np.where(np.isfinite(w), w, 0.0)

# ---- Fits ----
aPL,bPL,yPL,rssPL,aiccPL   = fit_powerlaw(xl, yl, wl=w)
aEXP,bEXP,yEXP,rssEXP,aiccEXP = fit_exponential(lam[sel], yl, wl=w)
bPLK = fit_broken_powerlaw(xl, yl, kmin=2, kmax=12, wl=w)

# ΔAICc
A = dict(PowerLaw=aiccPL, Exponential=aiccEXP, BrokenPL=bPLK["AICc"])
best_name = min(A, key=A.get)

print(f"Fit on k={sel.min()+1}..{sel.max()+1} (N={len(sel)})")
print(f" Power-law:   slope b={bPL:+.3f},  AICc={aiccPL:.1f}")
print(f" Exponential: slope b={bEXP:+.3e}, AICc={aiccEXP:.1f} (on log10E vs λ)")
print(f" Broken PL:   cut@k={sel[0]+bPLK['cut']+1}, AICc={bPLK['AICc']:.1f}")
print(" Model ranking (lower is better):")
for name,aic in sorted(A.items(), key=lambda kv: kv[1]):
    print(f"  - {name}: AICc={aic:.1f}")
print(" ΔAICc vs best:")
for name,aic in sorted(A.items(), key=lambda kv: kv[1]):
    print(f"  - {name}: {aic - A[best_name]:+.1f}")

# ---- Plot (log10 λ vs log10 E) with SEM shading & fits ----
fig, ax = plt.subplots(figsize=(6,4.5), dpi=150)

# SEM shading on log scale: show central line at log10(Emean), band from log10(E±SEM) (clip to positive)
Elo = np.clip(Emean - Esem, 1e-12, None)
Ehi = Emean + Esem
ax.fill_between(np.log10(lam), np.log10(Elo), np.log10(Ehi),
                alpha=0.25, linewidth=0, label="SEM band (all k)")

# Selected band points
ax.plot(xl, yl, '.', ms=4, label="Selected modes")

# Overlays:
ax.plot(xl, yPL, '-', lw=2, label=f"Power-law fit (b={bPL:+.2f})")
ax.plot(np.log10(lam[sel]), yEXP, '--', lw=2, label="Exponential fit")

# Broken PL
cut_glob = sel[0] + bPLK["cut"]
ax.plot(xl, bPLK["yhat"], '-', lw=2, label=f"Broken PL (cut k={cut_glob+1})")

ax.set_xlabel(r'$\log_{10}\,\lambda$')
ax.set_ylabel(r'$\log_{10}\,E$')
ax.set_title('Group spectrum with model fits')
ax.legend(frameon=False)
ax.grid(True, alpha=0.2)

outpng = gdir/"group_spectrum_model_fits.png"
plt.tight_layout()
plt.savefig(outpng, bbox_inches="tight")
plt.close()
print("Saved figure:", outpng)
PY