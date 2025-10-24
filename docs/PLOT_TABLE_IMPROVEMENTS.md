# Plot and Table Improvements Summary

This document describes the improvements made to the plotting and table generation pipeline.

## Changes Made

### 1. New Supplemental Tables Script

Created `python/generate_supplemental_tables.py` which generates:

#### a) NRMSE Pivot Tables by Derivative Order
- **Output**: One table per derivative order (0-7)
- **Format**: Methods × Noise Levels → NRMSE values
- **Sorting**: Methods sorted by average NRMSE (best to worst)
- **Columns**:
  - Noise levels: 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 2e-2, 5e-2
  - Grand Total: Average across all noise levels
- **Location**:
  - CSV: `build/tables/supplemental/nrmse_order_{0-7}.csv`
  - LaTeX: `build/tables/supplemental/nrmse_order_{0-7}.tex`

#### b) Average Timing Table
- **Output**: Single table showing average execution time per method
- **Sorting**: Methods sorted by execution time (fastest to slowest)
- **Columns**:
  - Method name
  - Average Time (s)
  - Data Points (number of measurements)
- **Location**:
  - CSV: `build/tables/supplemental/average_timing.csv`
  - LaTeX: `build/tables/supplemental/average_timing.tex`

### 2. Updated Plotting Script

Modified `python/generate_comprehensive_plots.py`:

#### a) Fixed Y-Axis Scale
- **Change**: Set y-axis limits to 0-1.0 for all line plots
- **Rationale**: Enables easy comparison across methods and derivative orders
- **Affected plots**:
  - Per-method noise sensitivity plots
  - Per-order method comparison plots

#### b) Dual Format Output
- **Change**: Generate both PDF and PNG for all supplemental plots
- **Formats**:
  - PDF: High-quality vector graphics for publication
  - PNG: Raster format for quick viewing and web use
- **Affected plots**:
  - Per-method heatmaps (28 methods × 2 formats = 56 files)
  - Per-method line plots (28 methods × 2 formats = 56 files)
  - Per-order comparisons (8 orders × 2 formats = 16 files)
- **Total**: 128 plot files (64 plots in 2 formats each)

### 3. Updated Build Scripts

Modified `scripts/04_generate_tables.sh`:
- Added call to `python/generate_supplemental_tables.py`
- Added verification for supplemental tables directory

## Output Summary

### Tables Generated
- **Publication tables**: `build/tables/publication/` (existing)
- **Supplemental tables**: `build/tables/supplemental/` (new)
  - 8 NRMSE pivot tables (one per derivative order)
  - 1 average timing table
  - Each table in both CSV and LaTeX formats

### Plots Generated
- **Per-method heatmaps**: 28 methods × 2 formats = 56 files
- **Per-method line plots**: 28 methods × 2 formats = 56 files
- **Per-order comparisons**: 8 orders × 2 formats = 16 files
- **Total**: 128 plot files

## Usage

### Generate Supplemental Tables Only
```bash
uv run python python/generate_supplemental_tables.py
```

### Generate All Tables (Publication + Supplemental)
```bash
./scripts/04_generate_tables.sh
```

### Generate All Plots
```bash
uv run python python/generate_comprehensive_plots.py
# or
./scripts/03_generate_figures.sh
```

### Full Pipeline
```bash
./scripts/build_all.sh
```

## File Locations

```
build/
├── tables/
│   ├── publication/          # Main paper tables
│   └── supplemental/         # NEW: Supplemental tables
│       ├── nrmse_order_0.csv
│       ├── nrmse_order_0.tex
│       ├── ...
│       ├── nrmse_order_7.csv
│       ├── nrmse_order_7.tex
│       ├── average_timing.csv
│       └── average_timing.tex
└── figures/
    ├── publication/          # Main paper figures
    └── supplemental/
        ├── per_method/       # 28 methods × 2 plot types × 2 formats
        │   ├── GP-Julia-AD_heatmap.pdf
        │   ├── GP-Julia-AD_heatmap.png
        │   ├── GP-Julia-AD_noise_sensitivity.pdf
        │   ├── GP-Julia-AD_noise_sensitivity.png
        │   └── ...
        └── per_order/        # 8 orders × 2 formats
            ├── order_0_method_comparison.pdf
            ├── order_0_method_comparison.png
            └── ...
```

## Key Features

### NRMSE Tables
- **Comprehensive**: Cover all 8 derivative orders (0-7)
- **Sorted**: Methods ordered by performance (best first)
- **Complete**: Include all noise levels tested
- **Summary**: Grand Total column shows average performance

### Timing Table
- **Practical**: Helps identify computational efficiency
- **Sorted**: Fastest methods listed first
- **Validated**: Based on actual execution times from benchmarks
- **Statistics**: Central-FD fastest (~84 μs), AAA-JAX-Adaptive-Wavelet slowest (~8.5 s)

### Plot Improvements
- **Standardized scale**: 0-1 y-axis for easy comparison
- **Reference lines**: Visual guides at 0.1, 0.5, 1.0 nRMSE
- **Dual formats**: PDF for publication, PNG for quick viewing
- **Color-coded**: Red-Yellow-Green heatmap (bad to good)

## Notes

- All tables are auto-generated from `build/results/comprehensive/comprehensive_summary.csv`
- Tables update automatically when re-running the benchmark
- Both LaTeX and CSV formats provided for flexibility
- PNG format at 300 DPI for high-quality raster output
