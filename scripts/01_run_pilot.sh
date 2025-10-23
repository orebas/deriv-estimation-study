#!/bin/bash
# 01_run_pilot.sh - Run minimal pilot study

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================"
echo "STEP 1: Running Pilot Study"
echo "========================================"

cd "$REPO_ROOT"

echo "Configuration is loaded from config.toml"
echo "(pilot_study section)"
echo ""

# Run pilot study
echo "Running pilot study..."
julia --project=. --startup-file=no src/pilot_study.jl

# Check results
if [ -f "build/results/pilot/pilot_results.csv" ]; then
    echo ""
    echo "✓ Pilot study complete!"
    echo "  Results: build/results/pilot/pilot_results.csv"
    wc -l build/results/pilot/pilot_results.csv

    if [ -f "build/results/pilot/pilot_summary.csv" ]; then
        echo "  Summary: build/results/pilot/pilot_summary.csv"
    fi
else
    echo "✗ ERROR: Pilot results not found!"
    exit 1
fi

echo "========================================"
