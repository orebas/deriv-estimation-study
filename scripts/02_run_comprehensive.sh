#!/bin/bash
# 02_run_comprehensive.sh - Run comprehensive benchmark study

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================"
echo "STEP 2: Running Comprehensive Study"
echo "========================================"

cd "$REPO_ROOT"

# Configuration (can be overridden by environment)
export DATA_SIZE="${DATA_SIZE:-51}"
export MAX_DERIV="${MAX_DERIV:-5}"
export NUM_TRIALS="${NUM_TRIALS:-10}"

echo "Configuration:"
echo "  Data size: $DATA_SIZE"
echo "  Max derivative order: $MAX_DERIV"
echo "  Number of trials: $NUM_TRIALS"
echo ""
echo "NOTE: This may take significant time depending on configuration."
echo ""

# Run comprehensive study
echo "Running comprehensive benchmark..."
julia --project=. --startup-file=no src/comprehensive_study.jl

# Check results
if [ -f "build/results/comprehensive/comprehensive_results.csv" ]; then
    echo ""
    echo "✓ Comprehensive study complete!"
    echo "  Results: build/results/comprehensive/comprehensive_results.csv"
    wc -l build/results/comprehensive/comprehensive_results.csv

    if [ -f "build/results/comprehensive/comprehensive_summary.csv" ]; then
        echo "  Summary: build/results/comprehensive/comprehensive_summary.csv"
    fi
else
    echo "✗ ERROR: Comprehensive results not found!"
    exit 1
fi

echo "========================================"
