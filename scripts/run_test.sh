#!/bin/bash
# run_test.sh - Run fast test study for method validation
#
# Test configuration:
# - 1 trial (instead of 10)
# - 2 ODEs: Lotka-Volterra, Van der Pol
# - 2 noise levels: 1e-4, 1e-3
# - Completes in ~10-15 minutes
#
# Use this for quick sanity checks when adding new methods.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================"
echo "QUICK TEST STUDY"
echo "========================================"
echo ""
echo "This is a fast sanity check with reduced parameters:"
echo "  - 1 trial (vs 10 in comprehensive)"
echo "  - 2 ODEs: Lotka-Volterra, Van der Pol"
echo "  - 2 noise levels: 1e-4, 1e-3"
echo "  - Total configs: 4 (vs 180 in comprehensive)"
echo ""
echo "Expected runtime: ~10-15 minutes"
echo ""

cd "$REPO_ROOT"

# Check environments
if [ ! -f "Manifest.toml" ] || [ ! -d "python/.venv" ]; then
    echo "✗ ERROR: Environments not set up!"
    echo "  Please run: ./scripts/setup.sh"
    exit 1
fi

# Run test study with parallelization
echo "Running test study with 4 threads..."
julia --project=. --threads=4 --startup-file=no src/test_study.jl

# Check results
if [ -f "build/results/comprehensive/test_results.csv" ]; then
    echo ""
    echo "✓ Test study complete!"
    echo "  Results: build/results/comprehensive/test_results.csv"
    wc -l build/results/comprehensive/test_results.csv

    if [ -f "build/results/comprehensive/test_summary.csv" ]; then
        echo "  Summary: build/results/comprehensive/test_summary.csv"
    fi

    if [ -f "build/results/comprehensive/test_failure_report.csv" ]; then
        echo "  Failures: build/results/comprehensive/test_failure_report.csv"
    fi
else
    echo "✗ ERROR: Test results not found!"
    exit 1
fi

echo ""
echo "========================================"
echo "TEST COMPLETE"
echo "========================================"
echo ""
echo "Output files saved to: build/results/comprehensive/"
echo "  - test_results.csv (detailed results)"
echo "  - test_summary.csv (aggregated statistics)"
echo "  - test_failure_report.csv (failure analysis)"
echo ""
echo "NOTE: This script only runs the study and generates CSV files."
echo "      It does NOT generate figures, tables, or compile the paper."
echo ""
echo "To run the full pipeline:"
echo "  ./scripts/build_all.sh"
echo "========================================"
