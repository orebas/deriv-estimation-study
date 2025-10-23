# Repository Reorganization

## Overview

This repository is being reorganized for reproducible computational science. The goal is to have:
- Clean separation between source code and generated data
- One method per file for easy auditing
- Automated regeneration from scratch
- No generated data committed to git

## Current Status

### Phase 1: Structure Setup ✓ COMPLETED

**Completed:**
- Created new directory structure
- Set up UV-managed Python virtual environment (.venv/)
- Configured local Julia project environment
- Added .gitkeep files for empty directories
- Updated .gitignore for new structure

**New Directory Structure:**
```
.
├── .venv/                      # UV-managed Python environment
├── methods/                    # Method implementations (one per file)
│   ├── julia/
│   │   ├── splines/
│   │   ├── gp/
│   │   ├── spectral/
│   │   ├── finite_diff/
│   │   ├── adaptive/
│   │   └── filtering/
│   └── python/
│       ├── splines/
│       ├── gp/
│       ├── spectral/
│       ├── finite_diff/
│       ├── adaptive/
│       └── filtering/
├── benchmark/                  # Data generation and benchmarking
├── paper/                      # Paper source code
├── build/                      # ALL generated data (starts EMPTY)
│   ├── data/
│   │   ├── input/
│   │   └── output/
│   ├── results/comprehensive/
│   ├── figures/
│   │   ├── publication/
│   │   └── supplemental/
│   └── tables/
├── build_tex/                  # LaTeX compilation (starts EMPTY)
├── scripts/                    # Build automation
└── docs/                       # Documentation
```

**Environment Setup:**

Python (UV-managed):
```bash
# Activate environment
source .venv/bin/activate

# Dependencies managed via pyproject.toml and requirements.lock
```

Julia (local project):
```bash
# Use project environment
julia --project=.

# Dependencies in Project.toml, resolved versions in Manifest.toml (gitignored)
```

## Next Phases

### Phase 2: Method Extraction
- Split python/python_methods.py into individual files by category (~30 methods)
- Split src/julia_methods.jl into individual files by category (~15 methods)
- Create methods/{julia,python}/common.{jl,py} for shared utilities
- Test each method after extraction

### Phase 3: Benchmark Runner Refactor
- Move src/{ground_truth,noise_model}.jl to benchmark/
- Create benchmark/run_benchmarks.jl that imports from methods/
- Update output paths to build/data/
- Move hyperparameters to benchmark/hyperparameters/

### Phase 4: Build Scripts
- Create scripts/01_generate_data.sh through 05_compile_paper.sh
- Create scripts/build_all.sh (runs 01-05 in sequence)
- Create scripts/clean.sh (removes build/, build_tex/)
- Ensure no hardcoded base directory names

### Phase 5: Paper Integration
- Move report/ to paper/
- Update paper/generate_figures.py paths
- Update paper/comprehensive_report.tex includes

### Phase 6: Documentation
- Update docs/README.md with new structure
- Document build process
- Add method documentation templates

## Design Principles

1. **Reproducibility**: Fresh clone → `./scripts/build_all.sh` → complete paper
2. **Portability**: No hardcoded paths, works regardless of repo name
3. **Auditability**: One method per file, clear organization
4. **Clean git**: Only source code committed, all outputs regenerable

## Using the New Structure

After reorganization is complete:

```bash
# Fresh start
git clone <repo>
cd <repo>

# Set up environments
uv venv .venv
source .venv/bin/activate
uv pip install numpy scipy scikit-learn matplotlib pandas autograd jax jaxlib

julia --project=. --startup-file=no -e 'using Pkg; Pkg.instantiate()'

# Build everything
./scripts/build_all.sh

# Or step by step
./scripts/01_generate_data.sh
./scripts/02_run_benchmarks.sh
./scripts/03_collate_results.sh
./scripts/04_generate_figures.sh
./scripts/05_compile_paper.sh

# Clean generated files
./scripts/clean.sh
```
