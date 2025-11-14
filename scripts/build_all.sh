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

# Step 0: Check environments
echo "Checking environments..."
ENV_OK=1

# Check Julia Manifest.toml
if [ ! -f "Manifest.toml" ]; then
    echo "✗ Julia environment not set up (Manifest.toml missing)"
    ENV_OK=0
fi

# Check Python venv
if [ ! -d "python/.venv" ]; then
    echo "✗ Python environment not set up (python/.venv missing)"
    ENV_OK=0
fi

if [ "$ENV_OK" = "0" ]; then
    echo ""
    echo "========================================"
    echo "ERROR: Environments not configured!"
    echo "========================================"
    echo ""
    echo "Please run the setup script first:"
    echo "  ./scripts/setup.sh"
    echo ""
    echo "This will:"
    echo "  - Install Julia packages from Project.toml"
    echo "  - Create Python virtual environment with uv"
    echo "  - Install Python packages from pyproject.toml"
    echo ""
    exit 1
fi

echo "✓ Environments configured"
echo ""

# Step 0a: Clean previous build (optional)
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

# Step 6: Flatten TeX for LLM parsing
if [ "${SKIP_FLATTEN:-0}" != "1" ]; then
    bash "$SCRIPT_DIR/06_flatten_tex.sh"
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
if [ -d "build/flattened" ]; then
    echo "  Flattened (for LLM parsing):"
    if [ -f "build/flattened/paper_flattened.tex" ]; then
        echo "    ✓ build/flattened/paper_flattened.tex"
    fi
    if [ -f "build/flattened/paper_text.txt" ]; then
        echo "    ✓ build/flattened/paper_text.txt"
    fi
fi
echo ""
echo "Useful commands:"
echo "  Clean build artifacts:  ./scripts/clean.sh"
echo "  Setup environments:     ./scripts/setup.sh"
echo "  Skip specific steps:    SKIP_PILOT=1 SKIP_COMPREHENSIVE=1 SKIP_FIGURES=1 SKIP_TABLES=1 SKIP_PAPER=1 SKIP_FLATTEN=1"
echo "========================================"
