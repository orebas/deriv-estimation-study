# Comprehensive Study Dataset - FROZEN

**Date Frozen:** October 29, 2024
**Reason:** Paper finalization - dataset is stable and results are finalized

## Purpose of Freeze

This dataset represents the final, stable results of the comprehensive derivative estimation benchmark study. It has been committed to the repository (despite normally being in .gitignore) for the following reasons:

1. **Paper Finalization:** The paper is being prepared for submission and requires a stable, reproducible dataset
2. **Result Stability:** After extensive testing and method tuning, these results represent the final benchmark
3. **Reproducibility:** Future readers of the paper can access the exact dataset used for analysis
4. **Workflow Checkpoint:** This freeze allows us to regenerate figures/tables without re-running the expensive comprehensive study

## Dataset Contents

- `comprehensive_results.csv` (4.0 MB): Individual trial results for all methods, orders, noise levels, and ODE systems
- `comprehensive_summary.csv` (731 KB): Aggregated statistics (mean, std, min, max) per method/order/noise configuration
- `failure_report.csv` (34 KB): Documentation of which methods failed on which configurations
- `predictions/` (178 MB): Raw prediction arrays from each method for detailed analysis

## Key Statistics

- **Methods evaluated:** 23
- **Derivative orders:** 0-7 (8 levels)
- **Noise levels:** 6 levels (1e-8 to 2e-2)
- **ODE systems:** 3 (Lotka-Volterra, Van der Pol, Lorenz)
- **Trials per configuration:** 10
- **Total experimental cells:** 2,970 summary rows, 29,668 raw result rows

## Regeneration

If you need to regenerate this dataset (e.g., after method changes):

```bash
# WARNING: This takes ~65 minutes
SKIP_PILOT=1 ./scripts/02_run_comprehensive.sh
```

This will overwrite the frozen dataset. Consider backing it up first if you want to preserve the original results.

## Study Configuration

The study was run with the following configuration (from `config.toml`):

- Julia threads: 7
- Data size: 251 points
- Max derivative order: 7
- Trials per config: 10
- Enabled ODE systems: ["lotka_volterra", "van_der_pol", "lorenz"]
- Noise levels: [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 2e-2]

## Notes

- The predictions directory contains 184 files (binary/CSV predictions for detailed plotting)
- Some methods have partial coverage (not all orders/noise levels) - see failure_report.csv
- nRMSE values >10 are capped in some visualizations but stored raw in the CSV files
