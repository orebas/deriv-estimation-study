#!/bin/bash
# build_all.sh - Master build script that runs entire pipeline

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================"
echo "DERIVATIVE ESTIMATION STUDY"
echo "Full Build Pipeline"
echo "========================================"
echo ""

cd "$REPO_ROOT"

# Step 0: Clean previous build (optional)
if [ "${CLEAN_FIRST:-0}" = "1" ]; then
    echo "Cleaning previous build..."
    bash "$SCRIPT_DIR/clean.sh"
    echo ""
fi

# Step 1: Run pilot study
if [ "${SKIP_PILOT:-0}" != "1" ]; then
    bash "$SCRIPT_DIR/01_run_pilot.sh"
    echo ""
fi

# Step 2: Run comprehensive study
if [ "${SKIP_COMPREHENSIVE:-0}" != "1" ]; then
    bash "$SCRIPT_DIR/02_run_comprehensive.sh"
    echo ""
fi

# Step 3: Generate figures
if [ "${SKIP_FIGURES:-0}" != "1" ]; then
    bash "$SCRIPT_DIR/03_generate_figures.sh"
    echo ""
fi

# Step 4: Generate tables
if [ "${SKIP_TABLES:-0}" != "1" ]; then
    bash "$SCRIPT_DIR/04_generate_tables.sh"
    echo ""
fi

# Step 5: Compile paper
if [ "${SKIP_PAPER:-0}" != "1" ]; then
    bash "$SCRIPT_DIR/05_compile_paper.sh"
    echo ""
fi

echo "========================================"
echo "BUILD COMPLETE!"
echo "========================================"
echo ""
echo "Generated artifacts in build/ directory:"
if [ -d "build/data" ]; then
    echo "  Data:"
    find build/data -name "*.json" -type f | wc -l | xargs echo "    JSON files:"
fi
if [ -d "build/results" ]; then
    echo "  Results:"
    find build/results -name "*.csv" -type f | wc -l | xargs echo "    CSV files:"
fi
if [ -d "build/figures" ]; then
    echo "  Figures:"
    find build/figures -type f \( -name "*.pdf" -o -name "*.png" \) | wc -l | xargs echo "    Image files:"
fi
if [ -d "build/tables" ]; then
    echo "  Tables:"
    find build/tables -type f \( -name "*.tex" -o -name "*.csv" \) | wc -l | xargs echo "    Table files:"
fi
if [ -d "build/paper" ]; then
    echo "  Paper:"
    if [ -f "build/paper/paper.pdf" ]; then
        echo "    ✓ build/paper/paper.pdf"
    fi
fi
echo ""
echo "To clean build artifacts: ./scripts/clean.sh"
echo "To skip specific steps: SKIP_PILOT=1 SKIP_COMPREHENSIVE=1 SKIP_FIGURES=1 SKIP_TABLES=1 SKIP_PAPER=1"
echo "========================================"
