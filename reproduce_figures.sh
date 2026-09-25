#!/usr/bin/env bash
set -euo pipefail

# Manuscript-level reproduction wrapper for:
# "The scale-spanning cortical geometry of language"
#
# This executes the canonical notebook from final precomputed outputs in pang_out/.
# It does NOT rerun raw fMRI preprocessing, language-model inference, or all upstream GLMs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

NOTEBOOK="${REPO_ROOT}/notebooks/01_reproduce_paper_figures.ipynb"
OUTPUT_DIR="${REPO_ROOT}/notebooks"
OUTPUT_NAME="01_reproduce_paper_figures_executed.ipynb"

if [[ ! -f "${NOTEBOOK}" ]]; then
    echo "ERROR: canonical notebook not found:"
    echo "  ${NOTEBOOK}"
    exit 1
fi

if [[ ! -d "${REPO_ROOT}/pang_out" ]]; then
    echo "ERROR: pang_out/ not found at repository root:"
    echo "  ${REPO_ROOT}/pang_out"
    echo "See PRECOMPUTED_OUTPUTS.md for required manuscript-level inputs."
    exit 1
fi

export EIGENMODE_REPO="${REPO_ROOT}"

echo "Repository: ${REPO_ROOT}"
echo "Notebook:   ${NOTEBOOK}"
echo
echo "Executing manuscript-level reproduction..."

jupyter nbconvert \
    --to notebook \
    --execute "${NOTEBOOK}" \
    --output-dir="${OUTPUT_DIR}" \
    --output="${OUTPUT_NAME}" \
    --ExecutePreprocessor.timeout=-1

echo
echo "Done."
echo "Executed notebook:"
echo "  ${OUTPUT_DIR}/${OUTPUT_NAME}"
