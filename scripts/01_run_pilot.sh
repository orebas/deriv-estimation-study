#!/bin/bash
# 01_run_pilot.sh - Run minimal pilot study

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================"
echo "STEP 1: Running Pilot Study"
echo "========================================"

cd "$REPO_ROOT"

# Configuration
export DSIZE=51
export MAX_DERIV=3
export NOISE=0.0
export PY_MAX_ORDER=3

echo "Configuration:"
echo "  Data size: $DSIZE"
echo "  Max derivative order: $MAX_DERIV"
echo "  Noise level: $NOISE"
echo ""

# Run pilot study
echo "Running minimal pilot..."
julia --project=. --startup-file=no src/minimal_pilot.jl

# Check results
if [ -f "build/results/pilot/minimal_pilot_results.csv" ]; then
    echo ""
    echo "✓ Pilot study complete!"
    echo "  Results: build/results/pilot/minimal_pilot_results.csv"
    wc -l build/results/pilot/minimal_pilot_results.csv
else
    echo "✗ ERROR: Pilot results not found!"
    exit 1
fi

echo "========================================"
