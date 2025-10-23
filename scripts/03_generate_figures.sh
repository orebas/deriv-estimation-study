#!/bin/bash
# 03_generate_figures.sh - Generate publication figures

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================"
echo "STEP 3: Generating Publication Figures"
echo "========================================"

cd "$REPO_ROOT"

# Check if comprehensive results exist
if [ ! -f "build/results/comprehensive/comprehensive_results.csv" ]; then
    echo "✗ ERROR: Comprehensive results not found!"
    echo "  Run ./scripts/02_run_comprehensive.sh first"
    exit 1
fi

# Run Python figure generation scripts
PYTHON_VENV="python/.venv/bin/python"

if [ ! -f "$PYTHON_VENV" ]; then
    echo "✗ ERROR: Python venv not found at $PYTHON_VENV"
    exit 1
fi

echo "Generating additional figures..."
$PYTHON_VENV python/generate_additional_figures.py

# Check output
if [ -d "build/figures/publication" ]; then
    echo ""
    echo "✓ Figures generated!"
    echo "  Output: build/figures/publication/"
    find build/figures/publication -name "*.pdf" -o -name "*.png" | wc -l | xargs echo "    Figure files:"
else
    echo "✗ ERROR: Figure output directory not found!"
    exit 1
fi

echo "========================================"
