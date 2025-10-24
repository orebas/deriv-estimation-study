# Repository Organization

This document describes the repository structure and file organization for the derivative estimation study.

## Directory Structure

```
derivative_estimation_study/
├── methods/              # Method implementations (the "what")
│   ├── julia/           # Julia method implementations
│   │   ├── common.jl    # Shared utilities and derivative computation
│   │   ├── adaptive/    # AAA rational approximation methods
│   │   ├── filtering/   # Savitzky-Golay and filtering methods
│   │   ├── gp/          # Gaussian process methods
│   │   ├── regularization/  # Trend filtering and regularization
│   │   └── spectral/    # Fourier and spectral methods
│   └── python/          # Python method implementations
│       ├── common.py    # Shared utilities and base classes
│       ├── adaptive/    # AAA with adaptive hyperparameters
│       ├── filtering/   # Filtering and smoothing methods
│       ├── gp/          # Gaussian process methods
│       ├── rational/    # Rational approximation methods
│       ├── spectral/    # Fourier and spectral methods
│       └── splines/     # Spline-based methods
│
├── src/                  # Study runners and shared infrastructure (the "how")
│   ├── julia_methods_integrated.jl   # Wrapper to call Julia methods
│   ├── pilot_study.jl                # Quick pilot study runner
│   ├── comprehensive_study.jl        # Full benchmark study runner
│   ├── ground_truth.jl               # Test function generation
│   ├── noise_model.jl                # Noise generation
│   └── hyperparameter_selection.jl   # Julia hyperparameter utilities
│
├── python/              # Python infrastructure and build scripts
│   ├── python_methods_integrated.py  # Wrapper to call Python methods
│   ├── generate_comprehensive_plots.py  # Per-method and per-order visualizations
│   ├── generate_paper_tables.py      # LaTeX table generation
│   ├── generate_additional_figures.py  # Supplemental figures
│   ├── hyperparameters.py            # Adaptive hyperparameter selection (shared)
│   ├── baryrat_jax.py                # JAX wrapper for AAA (shared)
│   ├── tvregdiff.py                  # TVRegDiff implementation (shared)
│   ├── jax_derivatives.py            # JAX automatic differentiation utilities
│   └── matern_optimized.py           # Optimized Matern kernel for GP
│
├── report/              # LaTeX source (pure source, no outputs)
│   ├── paper.tex        # Main paper structure and abstract
│   ├── sections/        # Paper content sections
│   │   ├── section1_introduction.tex
│   │   ├── section3_problem.tex
│   │   ├── section4_methodology.tex
│   │   ├── section5_methods.tex
│   │   ├── section6_results.tex
│   │   ├── section7_discussion.tex
│   │   ├── section8_recommendations.tex
│   │   ├── section9_limitations.tex
│   │   └── section10_conclusion.tex
│   └── references.bib   # Bibliography
│
├── build/               # All generated outputs (git-ignored)
│   ├── results/         # Raw data from studies
│   │   ├── pilot/       # Pilot study results (JSON)
│   │   └── comprehensive/  # Full study results
│   │       └── comprehensive_summary.csv  # ⭐ SINGLE SOURCE OF TRUTH
│   ├── figures/         # Generated plots
│   │   ├── publication/ # Main paper figures (PDF/PNG)
│   │   └── supplemental/  # Supplemental figures
│   ├── tables/          # Generated LaTeX tables
│   │   ├── publication/ # Main paper tables
│   │   └── supplemental/  # Supplemental tables
│   ├── tex/             # LaTeX build artifacts (.aux, .log, .out, .toc)
│   └── paper/           # Final compiled paper
│       └── paper.pdf    # 📄 FINAL OUTPUT
│
├── scripts/             # Build pipeline automation
│   ├── 01_run_pilot.sh          # Step 1: Quick validation
│   ├── 02_run_comprehensive.sh  # Step 2: Full benchmark
│   ├── 03_generate_figures.sh   # Step 3: Create all plots
│   ├── 04_generate_tables.sh    # Step 4: Create LaTeX tables
│   ├── 05_compile_paper.sh      # Step 5: Build PDF
│   ├── build_all.sh             # Run entire pipeline
│   └── clean.sh                 # Remove all build artifacts
│
├── docs/                # Documentation and analysis
│   ├── BUG_ANALYSIS_ZERO_DERIVATIVES.md  # Step-function bug investigation
│   ├── METHOD_API_SPEC.md       # Method interface specification
│   ├── METHOD_CATALOG.md        # Complete method catalog
│   └── PHASE2_PROGRESS.md       # Development progress tracking
│
├── Project.toml         # Julia project dependencies
├── Manifest.toml        # Julia dependency lock file
├── pyproject.toml       # Python project configuration
├── requirements.lock    # Python dependency lock file
├── uv.lock              # UV package manager lock file
└── README.md            # Main repository documentation
```

## Data Flow

The repository follows a **single-source-of-truth** data flow:

```
1. Study Runners (src/*.jl)
   └─> Generate noisy data + evaluate methods
       └─> Write JSON results to build/results/

2. Comprehensive Study
   └─> Aggregate all results into CSV
       └─> build/results/comprehensive/comprehensive_summary.csv  ⭐

3. Visualization Pipeline
   ├─> python/generate_comprehensive_plots.py
   │   └─> Reads CSV → Generates per-method and per-order plots
   │       └─> build/figures/publication/*.pdf
   │       └─> build/figures/supplemental/*.pdf
   │
   ├─> python/generate_paper_tables.py
   │   └─> Reads CSV → Generates LaTeX tables
   │       └─> build/tables/publication/*.tex
   │
   └─> python/generate_additional_figures.py
       └─> Reads CSV → Generates specialized figures
           └─> build/figures/publication/*.pdf

4. LaTeX Compilation (report/paper.tex)
   ├─> \input{build/tables/publication/*.tex}
   ├─> \includegraphics{build/figures/publication/*.pdf}
   └─> Build artifacts → build/tex/
   └─> Final PDF → build/paper/paper.pdf  📄
```

## Key Principles

### 1. **Pure Source vs. Generated Outputs**
- **Source directories** (`methods/`, `src/`, `python/`, `report/`, `scripts/`): Version controlled
- **Build directory** (`build/`): Git-ignored, fully regenerable

### 2. **Single Source of Truth**
- **Data**: `build/results/comprehensive/comprehensive_summary.csv`
- **Plots**: All read from the CSV, not from individual method results
- **Tables**: All generated from the CSV

### 3. **Method Organization**
- **Implementation**: In `methods/julia/` or `methods/python/` organized by category
- **Shared utilities**:
  - Julia: `methods/julia/common.jl`
  - Python: `methods/python/common.py`, `python/*.py` (for cross-category shared code)
- **Study infrastructure**: In `src/` (Julia) or `python/` (Python build scripts)

### 4. **Clear Separation**
- **Methods** (what): `methods/` - Pure algorithm implementations
- **Studies** (how): `src/` - How to run and evaluate methods
- **Infrastructure** (tools): `python/` - Shared utilities and build pipeline
- **Paper** (presentation): `report/` - LaTeX source
- **Outputs** (results): `build/` - All generated content

## How to Edit

### To modify paper text:
```bash
# Edit the relevant section file
vim report/sections/section6_results.tex

# Rebuild the paper
./scripts/05_compile_paper.sh

# Output: build/paper/paper.pdf
```

### To add a new method:
```julia
# 1. Create implementation
touch methods/julia/filtering/my_new_method.jl

# 2. Add to method list
vim src/julia_methods_integrated.jl

# 3. Re-run study
./scripts/02_run_comprehensive.sh
```

### To modify plots:
```python
# Edit plot generation
vim python/generate_comprehensive_plots.py

# Regenerate figures
./scripts/03_generate_figures.sh

# Output: build/figures/publication/*.pdf
```

### To clean and rebuild everything:
```bash
./scripts/clean.sh          # Remove all build artifacts
./scripts/build_all.sh      # Rebuild from scratch
```

## Build Pipeline Stages

1. **Pilot Study** (`01_run_pilot.sh`): Quick sanity check (~1 minute)
   - Tests subset of methods on simple test cases
   - Output: `build/results/pilot/*.json`

2. **Comprehensive Study** (`02_run_comprehensive.sh`): Full benchmark (~7-10 minutes)
   - All methods × all noise levels × all derivative orders × multiple trials
   - Output: `build/results/comprehensive/comprehensive_summary.csv` ⭐

3. **Generate Figures** (`03_generate_figures.sh`): Create all plots (~30 seconds)
   - Per-method heatmaps (noise × order)
   - Per-method line plots (noise sensitivity)
   - Per-order comparison plots (method rankings)
   - Output: `build/figures/publication/*.pdf` (62 plots total)

4. **Generate Tables** (`04_generate_tables.sh`): Create LaTeX tables (~10 seconds)
   - Summary tables (best methods per order)
   - Performance ranking tables
   - Method comparison matrices
   - Output: `build/tables/publication/*.tex`

5. **Compile Paper** (`05_compile_paper.sh`): Build final PDF (~20 seconds)
   - Includes auto-generated tables and figures
   - Output: `build/paper/paper.pdf` 📄

## Environment Setup

### Julia Dependencies
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Python Dependencies
```bash
uv sync                    # Using UV package manager
# OR
pip install -r requirements.lock
```

## Common Tasks

### Clean build and regenerate everything
```bash
./scripts/clean.sh && ./scripts/build_all.sh
```

### Quick edit-compile-view cycle for paper
```bash
vim report/sections/section6_results.tex
./scripts/05_compile_paper.sh
xdg-open build/paper/paper.pdf  # Linux
# open build/paper/paper.pdf    # macOS
```

### Re-run study after method changes
```bash
./scripts/02_run_comprehensive.sh  # Re-generate data
./scripts/03_generate_figures.sh   # Re-generate plots
./scripts/04_generate_tables.sh    # Re-generate tables
./scripts/05_compile_paper.sh      # Re-build paper
```

### Check what's been built
```bash
ls -R build/
```
