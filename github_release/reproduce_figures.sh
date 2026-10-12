#!/usr/bin/env bash
set -euo pipefail

# Reproduce the definitive sentence-onset manuscript results
# from curated precomputed numerical outputs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NOTEBOOK="${SCRIPT_DIR}/notebooks/01_reproduce_paper_figures.ipynb"
OUTPUT_DIR="${SCRIPT_DIR}/notebooks"

if [[ ! -f "${NOTEBOOK}" ]]; then
    echo "ERROR: canonical notebook missing: ${NOTEBOOK}"
    exit 1
fi

if [[ ! -d "${SCRIPT_DIR}/precomputed" ]]; then
    echo "ERROR: precomputed/ directory missing."
    exit 1
fi

cd "${SCRIPT_DIR}"

jupyter nbconvert \
    --to notebook \
    --execute "${NOTEBOOK}" \
    --output 01_reproduce_paper_figures_executed.ipynb \
    --output-dir "${OUTPUT_DIR}" \
    --ExecutePreprocessor.timeout=600

echo "PASS: manuscript-level reproduction completed."
