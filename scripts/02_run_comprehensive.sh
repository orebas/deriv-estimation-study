#!/bin/bash
# 02_run_comprehensive.sh - Run comprehensive benchmark study

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================"
echo "STEP 2: Running Comprehensive Study"
echo "========================================"

cd "$REPO_ROOT"

echo "Configuration is loaded from config.toml"
echo "(comprehensive_study section)"
echo ""
echo "NOTE: This may take significant time depending on configuration."
echo ""

# Run comprehensive study with parallelization
echo "Running comprehensive benchmark with 6 threads..."
julia --project=. --threads=6 --startup-file=no src/comprehensive_study.jl

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
