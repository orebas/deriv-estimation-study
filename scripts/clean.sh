#!/bin/bash
# clean.sh - Remove all generated build artifacts

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================"
echo "CLEAN: Removing build artifacts"
echo "========================================"

cd "$REPO_ROOT"

# Remove build directory contents (keep .gitkeep files)
if [ -d "build" ]; then
    echo "Cleaning build/ directory (includes results, figures, tables, tex, paper)..."
    find build -type f ! -name '.gitkeep' -delete
    echo "  ✓ Removed all generated files from build/"
fi

# Remove report/paper.pdf if it exists (should be in build/paper/ now)
if [ -f "report/paper.pdf" ]; then
    echo "Removing report/paper.pdf (output should be in build/paper/)..."
    rm -f "report/paper.pdf"
    echo "  ✓ Removed PDF from report/ directory"
fi

# Remove any scattered old generated data (from pre-reorganization workflow)
if [ -d "data" ]; then
    echo "Removing old data/ directory..."
    rm -rf "data"
    echo "  ✓ Removed old data/"
fi

if [ -d "results" ]; then
    echo "Removing old results/ directory..."
    rm -rf "results"
    echo "  ✓ Removed old results/"
fi

if [ -d "paper_figures" ]; then
    echo "Removing old paper_figures/ directory..."
    rm -rf "paper_figures"
    echo "  ✓ Removed old paper_figures/"
fi

if [ -f "build_paper.sh" ]; then
    rm -f "build_paper.sh"
    echo "  ✓ Removed old build_paper.sh"
fi

echo ""
echo "Clean complete! All generated artifacts removed."
echo "========================================"
