# Deprecated Materials

This folder contains files that have been superseded but are kept for reference.

## Plot Generation Scripts (SUPERSEDED)

**Replaced by:** `generate_linear_plots.py`

1. `generate_comprehensive_plots.py` - Original 20 plots with log10 scale
   - Problem: Log scale compressed meaningful differences
   - Deprecated: 2025-10-20

2. `generate_comprehensive_plots_filtered.py` - Filtered version, still log10
   - Problem: Still using log scale, included some bad methods
   - Deprecated: 2025-10-20

3. `generate_granular_plots.py` - 46 plots with NO averaging, log10 scale
   - Problem: Log scale still not ideal for nRMSE visualization
   - Deprecated: 2025-10-20

## Documentation (SUPERSEDED)

**Replaced by:** `CURRENT_STATUS.md` and `LINEAR_PLOTS_GUIDE.md`

1. `GRANULAR_PLOTS_GUIDE.md` - Documentation for granular plots
   - Superseded by linear scale approach
   - Deprecated: 2025-10-20

2. `FILTERED_PLOTS_SUMMARY.md` - Summary of filtered comprehensive plots
   - Superseded by linear scale plots
   - Deprecated: 2025-10-20

## LaTeX Files (OUTDATED)

1. `supplementary_plots.tex` - References old plot files
   - Points to `paper_figures/supplementary/` (old location)
   - Should reference `paper_figures/supplementary_linear/`
   - Deprecated: 2025-10-20

2. `plot_catalog.tex` - Comprehensive catalog of old plots
   - References non-existent plot files
   - Deprecated: 2025-10-20

## Analysis Scripts (OLD)

1. `analyze_results.py` - Early results analysis
2. `enhanced_analysis.py` - Enhanced version
3. `investigate_*.py` - Various investigation scripts
   - These were exploratory, findings incorporated into main analysis
   - Deprecated: 2025-10-20

---

## Current (Active) Files

- `generate_linear_plots.py` - **CURRENT** plot generation
- `generate_detailed_tables.py` - Table generation (still active)
- `LINEAR_PLOTS_GUIDE.md` - Guide to current plots
- `CURRENT_STATUS.md` - Project status and decisions
- `paper.tex` - Main paper
- `supplementary_data_tables_final.tex` - Data tables (active)

## Why Keep These?

- Historical reference
- May contain useful comments or insights
- Backup in case we need to regenerate old plots for comparison
- Documentation of evolution of visualization approach
