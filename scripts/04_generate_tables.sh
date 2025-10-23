#!/bin/bash
# 04_generate_tables.sh - Generate publication tables

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================"
echo "STEP 4: Generating Publication Tables"
echo "========================================"

cd "$REPO_ROOT"

# Check if comprehensive results exist
if [ ! -f "build/results/comprehensive/comprehensive_results.csv" ]; then
    echo "✗ ERROR: Comprehensive results not found!"
    echo "  Run ./scripts/02_run_comprehensive.sh first"
    exit 1
fi

# Run Python table generation script
PYTHON_VENV="python/.venv/bin/python"

if [ ! -f "$PYTHON_VENV" ]; then
    echo "✗ ERROR: Python venv not found at $PYTHON_VENV"
    exit 1
fi

echo "Generating LaTeX tables and plots..."
$PYTHON_VENV python/generate_paper_tables.py

# Check output
if [ -d "build/tables/publication" ]; then
    echo ""
    echo "✓ Tables generated!"
    echo "  Tables: build/tables/publication/"
    find build/tables/publication -name "*.tex" -o -name "*.csv" | wc -l | xargs echo "    Table files:"
fi

if [ -d "build/figures/publication" ]; then
    echo "  Figures: build/figures/publication/"
    find build/figures/publication -name "*.pdf" -o -name "*.png" | wc -l | xargs echo "    Figure files:"
fi

echo "========================================"
