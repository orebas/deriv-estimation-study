#!/bin/bash
# 05_compile_paper.sh - Compile LaTeX paper with auto-generated tables and figures

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================"
echo "STEP 5: Compiling LaTeX Paper"
echo "========================================"

cd "$REPO_ROOT"

# Check if figures and tables have been generated
if [ ! -d "build/figures/publication" ] || [ -z "$(ls -A build/figures/publication 2>/dev/null)" ]; then
    echo "✗ ERROR: Figures not found in build/figures/publication/"
    echo "  Run ./scripts/03_generate_figures.sh first"
    exit 1
fi

if [ ! -d "build/tables/publication" ] || [ -z "$(ls -A build/tables/publication 2>/dev/null)" ]; then
    echo "✗ ERROR: Tables not found in build/tables/publication/"
    echo "  Run ./scripts/04_generate_tables.sh first"
    exit 1
fi

# Create build_tex directory for LaTeX artifacts
mkdir -p build_tex

# Check for pdflatex
if ! command -v pdflatex &> /dev/null; then
    echo "✗ ERROR: pdflatex not found!"
    echo "  Please install TeX Live or MiKTeX"
    exit 1
fi

echo "Compiling paper.tex..."
echo ""

# Change to report directory for compilation
cd report

# Run pdflatex + bibtex + pdflatex + pdflatex (standard LaTeX build)
echo "First pass: pdflatex..."
pdflatex -interaction=nonstopmode paper.tex > /dev/null 2>&1
if [ ! -f "paper.pdf" ]; then
    echo "✗ ERROR: First pdflatex pass failed - PDF not created!"
    echo "  Check report/paper.log for details"
    exit 1
fi

echo "Running bibtex..."
bibtex paper > /dev/null 2>&1 || {
    echo "⚠ WARNING: bibtex failed (may be OK if no citations)"
}

echo "Second pass: pdflatex..."
pdflatex -interaction=nonstopmode paper.tex > /dev/null 2>&1
if [ ! -f "paper.pdf" ]; then
    echo "✗ ERROR: Second pdflatex pass failed - PDF not created!"
    echo "  Check report/paper.log for details"
    exit 1
fi

echo "Third pass: pdflatex..."
pdflatex -interaction=nonstopmode paper.tex > /dev/null 2>&1
if [ ! -f "paper.pdf" ]; then
    echo "✗ ERROR: Third pdflatex pass failed - PDF not created!"
    echo "  Check report/paper.log for details"
    exit 1
fi

# Check if PDF was created
if [ ! -f "paper.pdf" ]; then
    echo "✗ ERROR: paper.pdf not created!"
    echo "  Check report/paper.log for details"
    exit 1
fi

# Copy PDF to build directory
echo ""
echo "Copying paper.pdf to build/paper/..."
mkdir -p ../build/paper
cp paper.pdf ../build/paper/paper.pdf

# Move LaTeX artifacts to build_tex
echo "Moving LaTeX artifacts to build_tex/..."
mv paper.aux paper.log paper.out paper.toc paper.bbl paper.blg ../build_tex/ 2>/dev/null || true

echo ""
echo "✓ Paper compiled successfully!"
echo "  Output: build/paper/paper.pdf"
echo "  LaTeX artifacts: build_tex/"

# Count pages
if command -v pdfinfo &> /dev/null; then
    PAGES=$(pdfinfo ../build/paper/paper.pdf | grep "Pages:" | awk '{print $2}')
    echo "  Pages: $PAGES"
fi

echo "========================================"
