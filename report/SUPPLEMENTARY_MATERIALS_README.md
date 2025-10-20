# Supplementary Materials: Detailed Data Tables

## Overview

This directory contains comprehensive raw data tables to support all claims in the main paper.

## Files Generated

### 1. Supplementary Data Tables PDF
**File:** `supplementary_data_tables_final.pdf` (159 KB, 24 pages)

**Contents:**
- **Part I: Method × Noise Level Tables (Tables S1-S8)**
  - One table per derivative order (0-7)
  - Shows nRMSE for all 27 methods across all 7 noise levels
  - Allows detailed comparison at each specific order

- **Part II: Noise × Derivative Order Tables (Tables M1-M16)**
  - One table per full-coverage method (16 methods)
  - Shows nRMSE across all orders (0-7) and all noise levels
  - Reveals complete performance profile for each method

**Data Interpretation:**
- All values are mean nRMSE across 3 trials
- "---" indicates NaN/Inf or configuration not tested
- Values are unfiltered raw experimental results
- nRMSE < 0.1: Excellent | 0.1-0.5: Good | 0.5-1.0: Acceptable | >1.0: Poor | >10: Catastrophic

### 2. Source Data
**File:** `../results/comprehensive/comprehensive_summary.csv` (308 KB)
- 1,310 experimental data points
- Complete results for all method-order-noise combinations

### 3. Generation Scripts
**Files:**
- `generate_detailed_tables.py` - Automated LaTeX table generation from CSV
- `supplementary_data_tables_final.tex` - LaTeX source (698 lines)

## Key Tables for Reviewers

### Most Detailed Tables (Method × Noise, by Order):
- **Table S1** (Order 0): All methods at function interpolation task
- **Table S4** (Order 3): Critical transition point where AAA begins failing
- **Table S5** (Order 4): Separation of robust vs fragile methods
- **Table S8** (Order 7): Extreme challenge - only top methods survive

### Most Comprehensive Tables (Noise × Order, by Method):
- **Table M3** (Fourier-Interp): Best spectral method, shows graceful degradation
- **Table M4** (GP-Julia-AD): Top overall performer, consistent across all configs
- **Table M1** (AAA-HighPrec): Demonstrates catastrophic failure progression

## Data Validation

All tables generated automatically from experimental results with:
- ✓ No manual editing or filtering
- ✓ All 27 methods included (full + partial coverage)
- ✓ Complete 8×7 grid (orders 0-7, noise 1e-8 to 5e-2)
- ✓ Raw nRMSE values (not log-scaled or normalized)

## Answering Key Questions

**Q: "Which methods work at high noise?"**
→ See Tables S1-S8, rightmost columns (noise = 0.05)

**Q: "How does AAA fail catastrophically?"**
→ See Table M1 (AAA-HighPrec), watch values explode from order 3 onward

**Q: "Is GP-Julia-AD really consistent?"**
→ See Table M4, notice gradual increase (0.007 → 0.620) without catastrophic jumps

**Q: "What about partial-coverage methods?"**
→ See Tables S1-S2, methods with "---" entries show limited scope

**Q: "Noise sensitivity at a specific order?"**
→ Pick order, see corresponding Table S{N}, read across rows for each method

**Q: "Order progression for a specific method?"**
→ Pick method, see corresponding Table M{N}, read down rows for each noise level

## Build Instructions

To regenerate tables:
```bash
cd /home/orebas/derivative_estimation_study/report
python3 generate_detailed_tables.py > detailed_tables_content.tex
# Tables are auto-assembled into supplementary_data_tables_final.tex
pdflatex supplementary_data_tables_final.tex
```

## Future: Filtered Plots

Planned supplementary plots (requires fixing numpy/matplotlib compatibility):
1. Heatmap: Top 10 methods across orders (filtered, capped at nRMSE=10)
2. Noise sensitivity curves for top 5 methods at each order
3. Order progression at moderate noise (1e-3)
4. Per-method heatmaps (noise × order) for top 6 methods

Script ready: `generate_supplementary_plots.py` (awaiting dependency fix)

## Usage Recommendation

**For paper review:**
1. Cite main paper tables (Tables 2-7) for key claims
2. Point reviewers to this supplementary PDF for complete raw data
3. Specific challenges can be addressed by referencing exact table numbers

**For practitioners:**
1. Use Table S{N} (order-specific) to compare methods for your derivative order
2. Use Table M{N} (method-specific) to understand full capability of a method
3. Cross-reference with main paper's recommendations (Section 8)

## Statistics

- **Part I:** 8 tables × 27 methods × 7 noise levels = 1,512 cells
- **Part II:** 16 tables × 7 noise × 8 orders = 896 cells
- **Total data points shown:** ~2,400 nRMSE values
- **Coverage:** 1,310 / 1,512 theoretical max (87% coverage due to method limitations)

## License and Citation

Data generated from open-source benchmark study. If used, cite main paper.

---

**Generated:** October 20, 2025
**Data source:** Derivative Estimation Benchmark Study
**Contact:** See main paper for author information
