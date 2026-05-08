#!/usr/bin/env python
"""
Fix TR mismatches between wordrate regressor and eigenmode energy time series.

Loads original regressor and truncates to match the actual number of TRs
per run from the eigenmode energy arrays in:
    ~/eigenmode_fingerprints/pang_out/energy/{L,R}/run-XX.npy

Saves to: wordrate_per_run_fixed.json
"""

import json
import numpy as np
from pathlib import Path
from settings import PANG_OUT, HEMIS

# Input regressor dict
INPUT_JSON = Path(PANG_OUT) / "wordrate_per_run.json"
OUTPUT_JSON = Path(PANG_OUT) / "wordrate_per_run_fixed.json"

# Load existing wordrate
with open(INPUT_JSON, "r") as f:
    wordrate_orig = json.load(f)

fixed = {}

for hemi in HEMIS:
    energy_dir = Path(PANG_OUT) / "energy" / hemi
    for f in sorted(energy_dir.glob("run-*.npy")):
        run_id = f.stem  # 'run-15'
        energy = np.load(f)
        T_energy = energy.shape[1]

        if run_id not in wordrate_orig:
            continue

        w = np.asarray(wordrate_orig[run_id])
        T_wordrate = len(w)

        if T_wordrate == T_energy:
            print(f"✅ {run_id}: already matched (T={T_energy})")
            fixed[run_id] = w.tolist()
        elif T_wordrate > T_energy:
            print(f"🔪 {run_id}: truncating from {T_wordrate} → {T_energy}")
            fixed[run_id] = w[:T_energy].tolist()
        else:
            print(f"❗ {run_id}: regressor too short ({T_wordrate} < {T_energy}) — skipping")
            # optional: pad or drop; here we skip
            continue

# Save fixed dictionary
with open(OUTPUT_JSON, "w") as f:
    json.dump(fixed, f)

print(f"\n✅ Saved fixed regressor dictionary: {OUTPUT_JSON}")