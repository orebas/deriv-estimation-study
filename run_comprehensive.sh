#!/bin/bash
# Run comprehensive study with thread count from config.toml

# Get directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Read thread count from config.toml (default to 7 if not found)
THREADS=$(grep -A 2 "\[performance\]" config.toml | grep "julia_threads" | sed 's/.*= *//' | tr -d ' ')
THREADS=${THREADS:-7}

echo "Setting JULIA_NUM_THREADS=$THREADS (from config.toml)"
export JULIA_NUM_THREADS=$THREADS

echo "Running comprehensive study..."
julia --project=. src/comprehensive_study.jl "$@"
