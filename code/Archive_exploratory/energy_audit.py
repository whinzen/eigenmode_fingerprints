from pathlib import Path
import re

BASE = Path.home() / "eigenmode_fingerprints" / "code" / "energy"

FILES = [
    "energy_compute_per_run.py",
    "energy_spectrum_subject.py",
    "energy_all_collect.py",
    "energy_group_aggregate.py",
    "energy_fit_criticality.py",
    "energy_fit_group_criticality.py",
    "plot_group_fit_energy.py",
    "plot_group_fit_loglog_zoom.py",
    "slurm_energy_whole_dataset.sbatch",
]

PATTERNS = {
    "K / kmodes": [
        r"\bK\s*=\s*\d+",
        r"\bkmodes\s*=\s*\d+",
        r"\bN_MODES\s*=\s*\d+",
    ],
    "mode-0 handling": [
        r"mode_k\s*[<>!=]=?\s*0",
        r"drop.*mode",
        r"exclude.*mode",
        r"k\s*>\s*0",
    ],
    "energy definition": [
        r"A\s*\*\*\s*2",
        r"A_k\(t\)\^2",
        r"energy",
        r"np\.square",
    ],
    "log transforms": [
        r"log10",
        r"np\.log",
        r"np\.log10",
    ],
    "fit function": [
        r"linregress",
        r"polyfit",
        r"curve_fit",
        r"lstsq",
        r"power",
    ],
    "fit range": [
        r"kmin",
        r"kmax",
        r"fit_min",
        r"fit_max",
        r"mode_min",
        r"mode_max",
    ],
    "x-axis variable": [
        r"\blam\b",
        r"lambda",
        r"wavelength",
        r"mode_k",
        r"\bk\b",
    ],
    "hemisphere handling": [
        r"hemi",
        r"A_L",
        r"A_R",
        r"left",
        r"right",
    ],
    "outputs": [
        r"to_csv",
        r"savefig",
        r"np\.save",
        r"pickle",
    ],
}

def grep_patterns(text, patterns):
    hits = []
    for pat in patterns:
        found = re.findall(pat, text, flags=re.IGNORECASE)
        if found:
            hits.extend(found[:5])
    return hits

def main():
    for fname in FILES:
        path = BASE / fname
        print("\n" + "=" * 80)
        print(fname)
        print("=" * 80)

        if not path.exists():
            print("MISSING")
            continue

        text = path.read_text(errors="ignore")

        for label, pats in PATTERNS.items():
            hits = grep_patterns(text, pats)
            if hits:
                print(f"{label}: {hits}")

if __name__ == "__main__":
    main()