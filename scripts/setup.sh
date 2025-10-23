#!/bin/bash
# setup.sh - Set up Julia and Python environments
# Prerequisites: julia and uv must be installed

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================"
echo "ENVIRONMENT SETUP"
echo "========================================"
echo ""

cd "$REPO_ROOT"

# ============================================================================
# Check prerequisites
# ============================================================================

echo "Checking prerequisites..."

if ! command -v julia &> /dev/null; then
    echo "✗ ERROR: julia not found!"
    echo "  Please install Julia from https://julialang.org/downloads/"
    exit 1
fi

if ! command -v uv &> /dev/null; then
    echo "✗ ERROR: uv not found!"
    echo "  Please install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "  ✓ julia: $(julia --version | head -1)"
echo "  ✓ uv: $(uv --version)"
echo ""

# ============================================================================
# Set up Julia environment
# ============================================================================

echo "=========================================="
echo "Setting up Julia environment..."
echo "=========================================="
echo ""

# Check if Project.toml exists
if [ ! -f "Project.toml" ]; then
    echo "✗ ERROR: Project.toml not found!"
    exit 1
fi

# Instantiate Julia environment
echo "Running: julia --project=. -e 'using Pkg; Pkg.instantiate()'"
julia --project=. -e 'using Pkg; Pkg.instantiate()'

echo ""
echo "✓ Julia environment ready!"
echo ""

# Show installed packages
echo "Julia packages installed:"
julia --project=. -e 'using Pkg; Pkg.status()' | head -20
echo ""

# ============================================================================
# Set up Python environment
# ============================================================================

echo "=========================================="
echo "Setting up Python environment..."
echo "=========================================="
echo ""

cd python

# Check if pyproject.toml exists
if [ ! -f "pyproject.toml" ]; then
    echo "✗ ERROR: python/pyproject.toml not found!"
    exit 1
fi

# Create virtual environment with uv
echo "Running: uv venv"
uv venv

echo ""
echo "Running: uv pip install -e ."
uv pip install -e .

echo ""
echo "✓ Python environment ready!"
echo ""

# Show installed packages
echo "Python packages installed:"
.venv/bin/python -m pip list | head -20
echo ""

cd "$REPO_ROOT"

# ============================================================================
# Summary
# ============================================================================

echo "=========================================="
echo "SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "Environments created:"
echo "  Julia:  $(pwd)/ (Project.toml + Manifest.toml)"
echo "  Python: $(pwd)/python/.venv/"
echo ""
echo "Next steps:"
echo "  1. Run pilot study:        ./scripts/01_run_pilot.sh"
echo "  2. Run full pipeline:      ./scripts/build_all.sh"
echo "  3. Clean generated data:   ./scripts/clean.sh"
echo ""
echo "=========================================="
