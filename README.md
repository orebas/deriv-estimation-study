# Derivative Estimation Benchmark Study

A comprehensive benchmarking study comparing 42+ numerical methods for derivative estimation under various noise conditions and derivative orders (0-7).

## Purpose

This repository evaluates numerical methods for estimating derivatives from ODE trajectory data under varying noise levels and sampling densities. The primary objective is to assess which methods remain viable for high-order differentiation and to document objective, reproducible results.

## Quick Start

### Prerequisites

- **Julia** (1.9+) - [Download](https://julialang.org/downloads/)
- **uv** - Python package manager - [Install](https://docs.astral.sh/uv/getting-started/installation/)
- **pdflatex** (optional, for paper compilation) - Install via TeX Live or MiKTeX

### One-Time Setup

```bash
git clone https://github.com/orebas/deriv-estimation-study.git
cd deriv-estimation-study
./scripts/setup.sh
```

The setup script will automatically:
- Install Julia packages from `Project.toml`
- Create Python virtual environment in `python/.venv/`
- Install Python dependencies from `pyproject.toml`

### Run Full Pipeline

```bash
./scripts/build_all.sh
```

This executes the complete pipeline:
1. **Pilot study** (quick validation)
2. **Comprehensive study** (full benchmark → generates bottleneck CSV)
3. **Figure generation** (publication-quality plots)
4. **Table generation** (LaTeX table includes)
5. **Paper compilation** (final PDF output)

**Output**: `build/paper/paper.pdf` with complete results

## Repository Structure

```
deriv-estimation-study/
├── build/                      # All generated data (starts EMPTY)
│   ├── data/                   # Benchmark input/output data
│   ├── results/
│   │   └── comprehensive/      # ★ BOTTLENECK: CSV files
│   ├── figures/publication/    # Auto-generated figures
│   ├── tables/publication/     # Auto-generated LaTeX tables
│   └── paper/                  # Final compiled PDF
├── build_tex/                  # LaTeX artifacts (EMPTY)
├── methods/                    # 42 extracted method implementations
│   ├── julia/                  # Julia methods
│   └── python/                 # Python methods
├── scripts/                    # Build automation
│   ├── setup.sh               # ★ Run this first!
│   ├── build_all.sh           # Master pipeline
│   ├── clean.sh               # Remove generated data
│   └── 01-05_*.sh             # Individual pipeline steps
├── src/                        # Julia benchmark orchestration
├── python/                     # Python analysis & plotting
├── report/                     # LaTeX source
└── docs/                       # Documentation
```

## Individual Pipeline Steps

```bash
# Run specific steps
./scripts/01_run_pilot.sh           # Quick validation
./scripts/02_run_comprehensive.sh   # Full benchmark
./scripts/03_generate_figures.sh    # Create plots
./scripts/04_generate_tables.sh     # Generate LaTeX tables
./scripts/05_compile_paper.sh       # Compile PDF

# Skip steps
SKIP_PILOT=1 SKIP_FIGURES=1 ./scripts/build_all.sh

# Clean before build
CLEAN_FIRST=1 ./scripts/build_all.sh

# Clean all generated data
./scripts/clean.sh
```

## Method Inventory (42+ methods)

**Julia methods**:
- Spectral: Fourier interpolation, Chebyshev, trigonometric AD
- Gaussian Processes: RBF with automatic differentiation
- Filtering: Savitzky-Golay, Trend filtering (various orders)
- Rational: AAA algorithm (adaptive, high-precision variants)
- Splines: Cubic/quintic splines, smoothing splines
- Regularization: Total variation, Tikhonov
- Finite Differences: Central, forward, backward

**Python methods**:
- Spectral: Fourier, Fourier continuation, Chebyshev
- Gaussian Processes: RBF (isotropic/anisotropic), Matérn kernels
- Filtering: Butterworth, Savitzky-Golay, Kalman
- Splines: Cubic splines, smoothing splines with GCV
- Regularization: TVRegDiff
- Machine Learning: SVR-based smoothers

*Complete catalog: [`docs/METHOD_CATALOG.md`](docs/METHOD_CATALOG.md)*

## Architecture: Single Bottleneck Design

The pipeline uses a **single bottleneck** for data flow:

```
DATA COLLECTION
└── 02_run_comprehensive.sh
         ↓
    ★ BOTTLENECK ★
    build/results/comprehensive/
    ├── comprehensive_results.csv
    └── comprehensive_summary.csv
         ↓
PAPER GENERATION
├── 03_generate_figures.sh
├── 04_generate_tables.sh
└── 05_compile_paper.sh
```

**Key benefit**: Adding a new method only requires re-running the pipeline. Tables and figures update automatically.

## Reproducibility

This repository is designed for **reproducible computational science**:

1. **Clean separation**: Source code vs. generated data
2. **Version control**: Only source code committed to git
3. **Automated regeneration**: One command rebuilds everything
4. **Single bottleneck**: Clear data handoff between stages
5. **Portable**: Relative paths, no hardcoded directories
6. **Documented**: Comprehensive inline documentation

To reproduce on a new machine:
```bash
git clone https://github.com/orebas/deriv-estimation-study.git
cd deriv-estimation-study
./scripts/setup.sh      # Set up environments
./scripts/build_all.sh  # Generate everything
```

Output: `build/paper/paper.pdf` with identical results.

## Documentation

- [`docs/REORGANIZATION.md`](docs/REORGANIZATION.md) - Repository reorganization details
- [`docs/METHOD_CATALOG.md`](docs/METHOD_CATALOG.md) - Complete method catalog with 42 methods
- [`docs/METHOD_API_SPEC.md`](docs/METHOD_API_SPEC.md) - API specifications for adding methods
- [`docs/JULIA_API_SPEC.md`](docs/JULIA_API_SPEC.md) - Julia-specific API details
- [`docs/PHASE2_PROGRESS.md`](docs/PHASE2_PROGRESS.md) - Method extraction progress tracking

## Metrics and Evaluation

- **Errors**: RMSE and MAE against analytic ground truth
- **Endpoints**: Excluded from error computation
- **Non-finite predictions**: Masked before aggregation
- **Timing**: Wall-clock time per method execution
- **Coverage**: Methods tested across 7 noise levels × 8 derivative orders

## Performance Notes

- **Julia GPs**: Use y-centering, x z-scoring, noise floor, and escalating jitter for Cholesky stability
- **Python GPs**: scikit-learn with bounded optimization
- **Matérn kernels**: Limited support due to differentiability constraints for high orders

## Citations

If you use this benchmark in your research, please cite:

```bibtex
@software{derivative_estimation_benchmark,
  title={Derivative Estimation Benchmark Study},
  author={[Author Names]},
  year={2024},
  url={https://github.com/orebas/deriv-estimation-study}
}
```

## References

- **TVRegDiff**:
  - Julia: [`NoiseRobustDifferentiation.jl`](https://adrianhill.de/NoiseRobustDifferentiation.jl/dev/)
  - Python: [`stur86/tvregdiff`](https://github.com/stur86/tvregdiff)

## License

[To be determined]

## Contact

[Contact information]
