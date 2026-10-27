#!/usr/bin/env bash
set -euo pipefail

# Reproduce the final manuscript analyses from curated precomputed outputs.
# This does not rerun raw fMRI preprocessing or upstream model fitting.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EIGENMODE_REPO="${SCRIPT_DIR}"

python - "${SCRIPT_DIR}" <<'PY'
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

repo = Path(sys.argv[1])
release = repo / "github_release"

source = repo / "notebooks" / "01_reproduce_paper_figures.ipynb"
precomputed = release / "precomputed"
output = release / "01_reproduce_paper_figures_executed.ipynb"

if not source.is_file():
    raise FileNotFoundError(f"Final notebook not found: {source}")

if not precomputed.is_dir():
    raise FileNotFoundError(
        f"Curated precomputed outputs not found: {precomputed}"
    )

print(f"Notebook:    {source}", flush=True)
print(f"Precomputed: {precomputed}", flush=True)
print(f"Output:      {output}", flush=True)

nb = nbformat.read(source, as_version=4)

client = NotebookClient(
    nb,
    timeout=600,
    kernel_name="python3",
    resources={"metadata": {"path": str(release)}},
)

client.execute()
nbformat.write(nb, output)

print("\nSUCCESS: all notebook cells executed.", flush=True)
print(f"Executed notebook: {output}", flush=True)
print(f"Figures: {release / 'reproduced_figures'}", flush=True)
PY
