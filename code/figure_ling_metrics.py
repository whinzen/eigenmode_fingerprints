import numpy as np
import matplotlib.pyplot as plt

# Data
measures = ["shift", "pred_lin", "pred_ar", "subspace", "curvature"]
within = np.array([0.4437, 0.6155, 0.3926, 0.7587, 2.0014])
boundary = np.array([0.7366, 0.8625, 0.6645, 0.9701, 2.0244])
delta = boundary - within

t_vals = np.array([127.75, 105.84, 104.93, 64.14, 8.09])
p_vals = np.array([0.0, 0.0, 0.0, 0.0, 1.01e-15])

x = np.arange(len(measures))
width = 0.32

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: raw values
axes[0].bar(x - width/2, within, width, label="Within")
axes[0].bar(x + width/2, boundary, width, label="Boundary")

for i, (t, p) in enumerate(zip(t_vals, p_vals)):
    y = max(within[i], boundary[i])
    axes[0].text(i, y + 0.03 * y, f"t={t:.1f}", ha="center", fontsize=8)

axes[0].set_xticks(x)
axes[0].set_xticklabels(measures, rotation=30)
axes[0].set_title("A. Raw transition values")
axes[0].legend(frameon=False)
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# Panel B: effect sizes
axes[1].bar(x, delta)

for i, p in enumerate(p_vals):
    y = delta[i]
    if p < 1e-10:
        label = "***"
    elif p < 1e-5:
        label = "**"
    elif p < 0.01:
        label = "*"
    else:
        label = "n.s."
    axes[1].text(i, y + 0.02 * abs(y) + 0.005, label, ha="center", fontsize=12)

axes[1].axhline(0)
axes[1].set_xticks(x)
axes[1].set_xticklabels(measures, rotation=30)
axes[1].set_title("B. Boundary effect (Δ)")
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("linguistic_panel_with_stats.png", dpi=300)
plt.savefig("linguistic_panel_with_stats.pdf")
plt.show()