import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("~/eigenmode_fingerprints/pang_out").expanduser()

PAIRS = [
    ("shift", "pred_error_ar"),
    ("shift", "pred_error_subspace"),
    ("shift", "curvature"),
]

def normalize_curve(y):
    y = np.asarray(y, float)
    m = np.nanmax(np.abs(y))
    if not np.isfinite(m) or m == 0:
        return y
    return y / m

for base_metric, extra_metric in PAIRS:
    pair_name = f"{base_metric}__plus__{extra_metric}_resid"
    pair_dir = BASE / f"group_{pair_name}_glm"

    print("\n" + "=" * 70)
    print(pair_name)
    print("=" * 70)

    for hemi in ["L", "R"]:
        f_base = pair_dir / f"group_{pair_name}_{base_metric}_hemi-{hemi}_by_mode_subject_level.csv"
        f_extra = pair_dir / f"group_{pair_name}_{extra_metric}_resid_hemi-{hemi}_by_mode_subject_level.csv"

        if not f_base.exists() or not f_extra.exists():
            print(f"hemi {hemi}: missing files")
            continue

        d1 = pd.read_csv(f_base)
        d2 = pd.read_csv(f_extra)

        y1 = normalize_curve(d1["beta_mean"].values)
        y2 = normalize_curve(d2["beta_mean"].values)

        good = np.isfinite(y1) & np.isfinite(y2)
        if good.sum() < 3:
            print(f"hemi {hemi}: insufficient overlap")
            continue

        r = np.corrcoef(y1[good], y2[good])[0, 1]
        print(f"hemi {hemi}: normalized profile correlation = {r:.3f}")