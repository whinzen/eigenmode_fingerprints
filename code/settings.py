# ~/eigenmode_fingerprints/code/settings.py
from pathlib import Path

# --- Paths ---
BASE     = Path.home() / "eigenmode_fingerprints"
EMP_DIR  = BASE / "data" / "empirical"                 # surface data root
PANG_OUT = BASE / "pang_out"                           # outputs root
CSV_ANN  = BASE / "ds003643" / "annotation" / "EN" / "repunct" / "lppEN.csv"

# --- GLM / design ---
TR_SEC   = 2.0      # TR (s) for all runs (validated)
HRF_SEC  = 6.0      # canonical peak lag (use as simple boxcar shift)
MIN_TRS  = 50       # skip too short runs
RUN_OFFSET = 14     # CSV run_id=1..9 → BOLD run numbers = 15..23 (offset)
HEMIS   = ["L", "R"]

TR = 2.0  # seconds
WORDRATE_JSON = PANG_OUT / "wordrate_per_run.json"

# --- Eigenmode meta (k↦λ). This is produced by your spectrum step and shipped.
WAVELENGTH_TABLE = PANG_OUT / "group" / "wavelength_table.csv"

# --- Boundaries JSON (built from CSV) ---
BOUND_JSON = PANG_OUT / "boundaries_EN_from_csv.json"

# --- Per-run GLM output
GLM_PER_RUN_DIR = "glm_sentence/per_run"

