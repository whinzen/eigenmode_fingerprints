import pandas as pd
import numpy as np
from nilearn import plotting
from pathlib import Path
import matplotlib.pyplot as plt

# ====== CONFIGURATION ======
BASE = Path.home() / "eigenmode_fingerprints"
SCALING_CSV = BASE / "pang_out/group/group_temporal_scaling.csv"
SURFACE_DIR = BASE / "pang_surfaces/fsaverage5"
OUT_DIR = BASE / "pang_out/group/scaling_maps"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load surface files (GIFTI format)
SURF_L = SURFACE_DIR / "lh.white.surf.gii"
SURF_R = SURFACE_DIR / "rh.white.surf.gii"

# Load scaling data
df = pd.read_csv(SCALING_CSV)
V = len(df) // 2  # assume symmetric hemispheres

# Split into hemispheres
data = {
    "L": df.iloc[:V].reset_index(drop=True),
    "R": df.iloc[V:].reset_index(drop=True)
}

# Measures and plotting ranges
measures = {
    "slope_alpha": {
        "label": "1/f slope (\u03b1)",
        "vmin": -2,
        "vmax": 0,
        "cmap": "plasma"
    },
    "fractal_dim": {
        "label": "Fractal dim D_f",
        "vmin": 1.00,
        "vmax": 1.07,
        "cmap": "magma"
    },
    "slope_alpha_SEM": {
        "label": "SEM of \u03b1",
        "vmin": 0,
        "vmax": 0.1,
        "cmap": "Oranges"
    },
	"fractal_dim_SEM": {
    	"label": "SEM of D_f",
    	"vmin": 0.0004,
    	"vmax": 0.0008,
    	"cmap": "Oranges"
    }
}

# View settings
views = {
    "lateral": "lateral",
    "medial": "medial"
}

# Plotting function
def plot_stat_map(data, hemi, surf_mesh, stat_name, config):
    stat = data[hemi][stat_name].values
    hemi_full = {"L": "left", "R": "right"}[hemi]

    for view_label, view in views.items():
        title = f"Group {config['label']} - {hemi_full.upper()} ({view_label})"
        outfile = OUT_DIR / f"group_{stat_name}_{hemi}_{view_label}.png"

        fig = plotting.plot_surf_stat_map(
            surf_mesh=str(surf_mesh),
            stat_map=stat,
            hemi=hemi_full,
            view=view,
            colorbar=True,
            title=title,
            cmap=config["cmap"],
            bg_map=None,
            darkness=0.5,
            vmin=config["vmin"],
            vmax=config["vmax"],
            output_file=str(outfile),
        )
        print(f"✅ Saved: {outfile.name}")

# Run plotting loop
for stat_name, config in measures.items():
    for hemi, surf_mesh in zip(["L", "R"], [SURF_L, SURF_R]):
        plot_stat_map(data, hemi, surf_mesh, stat_name, config)