import pandas as pd
from pathlib import Path

BASE = Path("~/eigenmode_fingerprints/pang_out").expanduser()

PAIRS = [
    ("shift", "pred_error_ar"),
    ("shift", "pred_error_subspace"),
    ("shift", "curvature"),
]

for base_metric, extra_metric in PAIRS:
    pair_name = f"{base_metric}__plus__{extra_metric}_resid"
    pair_dir = BASE / f"group_{pair_name}_glm"

    print("\n" + "=" * 80)
    print(f"PAIR: {pair_name}")
    print("=" * 80)

    for hemi in ["L", "R"]:
        f_base = pair_dir / f"group_{pair_name}_{base_metric}_hemi-{hemi}_by_mode_subject_level.csv"
        f_extra = pair_dir / f"group_{pair_name}_{extra_metric}_resid_hemi-{hemi}_by_mode_subject_level.csv"

        if not f_base.exists():
            print(f"[missing] {f_base}")
            continue
        if not f_extra.exists():
            print(f"[missing] {f_extra}")
            continue

        d1 = pd.read_csv(f_base)
        d2 = pd.read_csv(f_extra)

        print(f"\n=== hemi {hemi} ===")
        print(
            f"{base_metric}: significant modes = {int((d1['sig_q05'] == 1).sum())} | "
            f"min p = {d1['p'].min():.3e} | "
            f"max |beta_mean| = {d1['beta_mean'].abs().max():.6g}"
        )
        print(
            f"{extra_metric}_resid: significant modes = {int((d2['sig_q05'] == 1).sum())} | "
            f"min p = {d2['p'].min():.3e} | "
            f"max |beta_mean| = {d2['beta_mean'].abs().max():.6g}"
        )

        # print first few rows for quick inspection
        print(f"\nTop rows: {base_metric}, hemi {hemi}")
        print(d1.head(10).to_string(index=False))

        print(f"\nTop rows: {extra_metric}_resid, hemi {hemi}")
        print(d2.head(10).to_string(index=False))