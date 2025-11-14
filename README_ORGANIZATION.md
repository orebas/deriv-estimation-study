# Repository Organization

This document describes the repository structure and file organization for the derivative estimation study.

## ⚠️ CRITICAL: Python Environment

**ALWAYS use `python/.venv/bin/python` or `./run_python.sh` for Python scripts in this project.**

DO NOT use system python, python3, or uv run python - they will fail with numpy version conflicts.

```bash
# ✓ CORRECT:
python/.venv/bin/python python/script.py
./run_python.sh python/script.py

# ✗ WRONG:
python3 python/script.py           # System python - broken
python python/script.py            # System python - broken
uv run python python/script.py     # Broken - encodings error
.venv/bin/python python/script.py  # Wrong venv location
```

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
│   ├── generate_paper_tables.py      # LaTeX table generation and main heatmap
│   ├── generate_additional_figures.py  # Supplemental figures
│   ├── generate_supplemental_heatmaps.py  # Order-specific heatmaps (0-5, 6-7)
│   ├── generate_high_order_heatmaps.py   # 3-panel high-order analysis
│   ├── flatten_tex.py                # Flatten TeX for LLM parsing
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
│   │   ├── section2_related_work.tex
│   │   ├── section3_methods.tex
│   │   ├── section4_design.tex
│   │   ├── section5_analysis.tex
│   │   ├── section6_conclusion.tex
│   │   ├── appendixA_method_catalog_complete.tex
│   │   └── appendixB_high_order.tex
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
│   │   ├── publication/tables/ # Main paper tables (*.tex)
│   │   └── supplemental/  # Supplemental tables
│   ├── flattened/       # Flattened paper for LLM parsing
│   │   ├── paper_flattened.tex  # Full TeX with includes expanded
│   │   └── paper_text.txt       # Simplified text version
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
│   ├── 06_flatten_tex.sh        # Step 6: Flatten TeX for LLMs
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
   │   └─> Reads CSV → Generates LaTeX tables + main heatmap
   │       └─> build/tables/publication/tables/*.tex
   │       └─> build/figures/publication/top_methods_heatmap.png
   │
   ├─> python/generate_supplemental_heatmaps.py
   │   └─> Reads CSV → Generates order-specific heatmaps
   │       └─> build/figures/supplemental/heatmap_orders_0to5.png
   │       └─> build/figures/supplemental/heatmap_orders_6to7.png
   │
   ├─> python/generate_high_order_heatmaps.py
   │   └─> Reads CSV → Generates 3-panel high-order analysis
   │       └─> build/figures/publication/heatmap_orders_6to7_by_noise_regime.png
   │
   └─> python/generate_additional_figures.py
       └─> Reads CSV → Generates specialized figures
           └─> build/figures/publication/*.pdf

4. LaTeX Compilation (report/paper.tex)
   ├─> \input{build/tables/publication/tables/*.tex}
   ├─> \includegraphics{build/figures/publication/*.pdf}
   └─> Build artifacts → build/tex/
   └─> Final PDF → build/paper/paper.pdf  📄

5. TeX Flattening (python/flatten_tex.py)
   ├─> Reads report/paper.tex + all included sections
   └─> Generates flattened versions for LLM parsing
       ├─> build/flattened/paper_flattened.tex (full TeX with includes expanded)
       └─> build/flattened/paper_text.txt (simplified text version)
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
vim report/sections/section5_analysis.tex

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
   - Output: `build/tables/publication/tables/*.tex`

5. **Compile Paper** (`05_compile_paper.sh`): Build final PDF (~20 seconds)
   - Includes auto-generated tables and figures
   - Output: `build/paper/paper.pdf` 📄

6. **Flatten TeX** (`06_flatten_tex.sh`): Create LLM-parseable version (~5 seconds)
   - Recursively expands all \input and \include commands
   - Removes or simplifies figures, tables, and non-text elements
   - Output: `build/flattened/paper_flattened.tex` and `build/flattened/paper_text.txt`

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
vim report/sections/section5_analysis.tex
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
