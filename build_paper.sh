#!/bin/bash
# Comprehensive Build Script for Derivative Estimation Study
# Usage:
#   ./build_paper.sh          - Regenerate figures/tables and compile paper
#   ./build_paper.sh --clean  - Delete all generated artifacts
#   ./build_paper.sh --help   - Show help

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Print with color
print_header() {
    echo -e "\n${BLUE}================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================${NC}\n"
}

print_step() {
    echo -e "${GREEN}>>> $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

# Regenerate all analysis data
regenerate_data() {
    print_header "REGENERATING COMPREHENSIVE ANALYSIS DATA"

    # Check for Julia
    if ! command -v julia &> /dev/null; then
        print_error "Julia not found"
        echo "  Install Julia from https://julialang.org/"
        exit 1
    fi

    print_step "Running comprehensive study..."
    print_warning "This may take 30+ minutes to several hours depending on your system"

    # Create results directory if needed
    mkdir -p results/comprehensive

    # Run Julia analysis with output
    julia src/comprehensive_study.jl 2>&1 | tee comprehensive_study.log

    if [ $? -ne 0 ]; then
        print_error "Comprehensive study failed"
        echo "  Check: comprehensive_study.log"
        tail -50 comprehensive_study.log
        exit 1
    fi

    # Verify outputs
    if [ ! -f "results/comprehensive/comprehensive_summary.csv" ]; then
        print_error "Expected output not found: comprehensive_summary.csv"
        exit 1
    fi

    if [ ! -f "results/comprehensive/comprehensive_results.csv" ]; then
        print_error "Expected output not found: comprehensive_results.csv"
        exit 1
    fi

    # Show summary
    SUMMARY_LINES=$(wc -l < results/comprehensive/comprehensive_summary.csv)
    RESULTS_LINES=$(wc -l < results/comprehensive/comprehensive_results.csv)

    print_step "Data regeneration complete"
    echo "  Summary: $SUMMARY_LINES lines"
    echo "  Results: $RESULTS_LINES lines"
}

# Clean mode
clean() {
    print_header "CLEANING GENERATED ARTIFACTS"

    print_step "Removing compiled paper..."
    cd report
    rm -f comprehensive_report.pdf comprehensive_report.aux comprehensive_report.log \
          comprehensive_report.bbl comprehensive_report.blg comprehensive_report.out \
          comprehensive_report.toc comprehensive_report_final.log

    print_step "Removing auto-generated LaTeX tables..."
    rm -f paper_figures/overall_performance_*.tex

    print_step "Removing generated figures..."
    rm -f paper_figures/publication/*.png paper_figures/publication/*.pdf

    print_step "Removing CSV exports..."
    rm -f paper_figures/*.csv
    rm -f paper_figures/regime_table_*.csv

    print_step "Removing supplemental figures..."
    rm -f paper_figures/supplemental/*.png paper_figures/supplemental/*.pdf

    print_step "Removing analysis cache files..."
    cd ..
    rm -f results/comprehensive/*.pkl

    echo -e "${GREEN}Clean complete!${NC}"
    exit 0
}

# Clean all mode (including raw data)
clean_all() {
    print_header "CLEANING ALL ARTIFACTS AND RAW DATA"

    # First do regular clean
    cd report
    print_step "Removing compiled paper..."
    rm -f comprehensive_report.pdf comprehensive_report.aux comprehensive_report.log \
          comprehensive_report.bbl comprehensive_report.blg comprehensive_report.out \
          comprehensive_report.toc comprehensive_report_final.log

    print_step "Removing auto-generated LaTeX tables..."
    rm -f paper_figures/overall_performance_*.tex

    print_step "Removing generated figures..."
    rm -f paper_figures/publication/*.png paper_figures/publication/*.pdf

    print_step "Removing CSV exports..."
    rm -f paper_figures/*.csv
    rm -f paper_figures/regime_table_*.csv

    print_step "Removing supplemental figures..."
    rm -f paper_figures/supplemental/*.png paper_figures/supplemental/*.pdf

    cd ..

    # Now remove raw data
    print_step "Removing raw analysis results..."
    rm -f data/output/*.json

    print_step "Removing comprehensive analysis results..."
    rm -f results/comprehensive/comprehensive_summary.csv
    rm -f results/comprehensive/comprehensive_results.csv

    print_step "Removing analysis cache files..."
    rm -f results/comprehensive/*.pkl

    print_step "Removing analysis logs..."
    rm -f comprehensive_study.log

    echo -e "${GREEN}Complete clean finished!${NC}"
    exit 0
}

# Show help
show_help() {
    cat << EOF
Derivative Estimation Study - Comprehensive Build Script

USAGE:
    ./build_paper.sh [OPTIONS]

OPTIONS:
    (none)             Regenerate all figures/tables and compile paper (default)
    --full             Complete rebuild: run analysis, regenerate figures, compile paper
    --regenerate-data  Only run comprehensive analysis (Julia + Python)
    --clean            Delete all generated artifacts (figures, tables, PDFs, CSVs)
    --clean-all        Delete generated artifacts AND raw analysis results
    --figures-only     Only regenerate figures and tables (skip paper compilation)
    --paper-only       Only compile paper (skip figure regeneration)
    --help             Show this help message

WHAT THIS SCRIPT DOES:

Normal Mode (no flags):
    1. Checks that results data exists (results/comprehensive/*.csv)
    2. Activates Python virtual environment
    3. Runs generate_updated_figures.py to create:
       - All publication figures (PNG + PDF)
       - All supplemental figures
       - Auto-generated LaTeX table files
       - CSV exports for external use
    4. Compiles comprehensive_report.tex to PDF (2 passes)
    5. Reports final PDF size and location

Full Mode (--full):
    1. Runs comprehensive analysis (Julia + Python benchmarks)
       - WARNING: This can take 30+ minutes to hours depending on system
       - Processes all noise levels × derivative orders × trials
    2. Then runs normal mode (figures + paper)

Regenerate Data Mode (--regenerate-data):
    - Only runs comprehensive_study.jl
    - Generates results/comprehensive/*.csv files
    - Does NOT regenerate figures or compile paper
    - Useful when testing method changes

Clean Mode (--clean):
    - Removes generated files:
      * PDFs, figures, tables, CSVs
      * LaTeX auxiliary files
      * Analysis caches
    - Leaves raw analysis results (data/output/*.json) untouched

Clean All Mode (--clean-all):
    - Removes EVERYTHING including:
      * All --clean targets
      * Raw analysis results (data/output/*.json)
      * Comprehensive results (results/comprehensive/*.csv)
    - Complete fresh start

REQUIREMENTS:
    - Julia (for --full or --regenerate-data)
    - Python 3 with venv at python/.venv
    - pdflatex (TeX Live)
    - results/comprehensive/*.csv (generated by --regenerate-data or --full)

EXAMPLES:
    # Complete rebuild from scratch (analysis + figures + paper)
    ./build_paper.sh --full

    # Rebuild figures and paper (use existing analysis results)
    ./build_paper.sh

    # Only regenerate analysis data
    ./build_paper.sh --regenerate-data

    # Just update figures
    ./build_paper.sh --figures-only

    # Clean everything except raw data
    ./build_paper.sh --clean

    # Complete clean slate
    ./build_paper.sh --clean-all

OUTPUT LOCATION:
    report/comprehensive_report.pdf

EOF
    exit 0
}

# Check prerequisites
check_prerequisites() {
    print_step "Checking prerequisites..."

    # Check for results data
    if [ ! -f "results/comprehensive/comprehensive_summary.csv" ]; then
        print_error "Missing results/comprehensive/comprehensive_summary.csv"
        echo "  Run your analysis scripts first to generate comprehensive results."
        exit 1
    fi

    if [ ! -f "results/comprehensive/comprehensive_results.csv" ]; then
        print_error "Missing results/comprehensive/comprehensive_results.csv"
        echo "  Run your analysis scripts first to generate comprehensive results."
        exit 1
    fi

    # Check for Python venv
    if [ ! -d "python/.venv" ]; then
        print_error "Python virtual environment not found at python/.venv"
        echo "  Create it with: python3 -m venv python/.venv"
        exit 1
    fi

    # Check for pdflatex
    if ! command -v pdflatex &> /dev/null; then
        print_error "pdflatex not found"
        echo "  Install TeX Live or MiKTeX"
        exit 1
    fi

    print_step "Prerequisites OK"
}

# Generate all figures and tables
generate_figures() {
    print_header "GENERATING FIGURES AND TABLES"

    print_step "Activating Python environment..."
    source python/.venv/bin/activate

    print_step "Running generate_updated_figures.py..."
    cd report
    python generate_updated_figures.py

    if [ $? -ne 0 ]; then
        print_error "Figure generation failed"
        exit 1
    fi

    print_step "Verifying outputs..."

    # Check publication figures
    REQUIRED_FIGS=(
        "paper_figures/publication/figure1_heatmap.png"
        "paper_figures/publication/figure2_small_multiples.png"
        "paper_figures/publication/figure4_pareto.png"
        "paper_figures/publication/gp_julia_ad_detail.png"
    )

    for fig in "${REQUIRED_FIGS[@]}"; do
        if [ ! -f "$fig" ]; then
            print_warning "Missing: $fig"
        fi
    done

    # Check auto-generated tables
    if [ ! -f "paper_figures/overall_performance_low_noise.tex" ]; then
        print_error "Missing auto-generated table: overall_performance_low_noise.tex"
        exit 1
    fi

    if [ ! -f "paper_figures/overall_performance_high_noise.tex" ]; then
        print_error "Missing auto-generated table: overall_performance_high_noise.tex"
        exit 1
    fi

    cd ..
    print_step "Figure generation complete"
}

# Compile paper
compile_paper() {
    print_header "COMPILING PAPER"

    cd report

    print_step "First pdflatex pass..."
    pdflatex -interaction=nonstopmode comprehensive_report.tex > /dev/null 2>&1

    if [ $? -ne 0 ]; then
        print_error "First LaTeX compilation failed"
        echo "  Check: report/comprehensive_report.log"
        tail -30 comprehensive_report.log
        exit 1
    fi

    print_step "Second pdflatex pass (resolving references)..."
    pdflatex -interaction=nonstopmode comprehensive_report.tex > comprehensive_report_final.log 2>&1

    if [ $? -ne 0 ]; then
        print_error "Second LaTeX compilation failed"
        echo "  Check: report/comprehensive_report.log"
        tail -30 comprehensive_report_final.log
        exit 1
    fi

    if [ ! -f "comprehensive_report.pdf" ]; then
        print_error "PDF not generated"
        exit 1
    fi

    # Get PDF info
    PDF_SIZE=$(ls -lh comprehensive_report.pdf | awk '{print $5}')
    PDF_PAGES=$(pdfinfo comprehensive_report.pdf 2>/dev/null | grep "Pages:" | awk '{print $2}' || echo "unknown")

    cd ..

    print_step "Paper compilation complete"
    echo "  Location: report/comprehensive_report.pdf"
    echo "  Size: $PDF_SIZE"
    echo "  Pages: $PDF_PAGES"
}

# Main execution
main() {
    # Parse arguments
    case "${1:-}" in
        --clean)
            clean
            ;;
        --clean-all)
            clean_all
            ;;
        --help|-h)
            show_help
            ;;
        --regenerate-data)
            print_header "DATA REGENERATION MODE"
            regenerate_data
            print_header "DATA REGENERATION COMPLETE"
            echo "  Next steps:"
            echo "    Run './build_paper.sh' to generate figures and compile paper"
            ;;
        --full)
            print_header "FULL BUILD MODE (Data + Figures + Paper)"
            regenerate_data
            echo ""
            generate_figures
            compile_paper
            print_header "FULL BUILD COMPLETE"
            ;;
        --figures-only)
            print_header "FIGURES-ONLY MODE"
            check_prerequisites
            generate_figures
            print_header "BUILD COMPLETE (figures only)"
            ;;
        --paper-only)
            print_header "PAPER-ONLY MODE"
            cd report
            if [ ! -f "paper_figures/overall_performance_low_noise.tex" ]; then
                print_error "Auto-generated tables missing. Run without --paper-only first."
                exit 1
            fi
            cd ..
            compile_paper
            print_header "BUILD COMPLETE (paper only)"
            ;;
        "")
            # Default: figures + paper (assumes data exists)
            print_header "BUILD MODE (Figures + Paper)"
            check_prerequisites
            generate_figures
            compile_paper
            print_header "BUILD COMPLETE"
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Run './build_paper.sh --help' for usage"
            exit 1
            ;;
    esac
}

main "$@"
