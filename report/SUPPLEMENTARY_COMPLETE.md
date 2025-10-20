# Supplementary Materials - Complete

## Issue Resolution

**Problem:** Was using system Python instead of the venv  
**Solution:** The venv at `/home/orebas/derivative_estimation_study/venv` has compatible packages  
**Fix:** Activated venv before running plot generation  

## Files to Open Now

### 1. 📊 Detailed Data Tables
**File:** `supplementary_data_tables_final.pdf` (25 pages, 159 KB)

**Contains:**
- **Tables S1-S8:** Method × Noise (one per derivative order 0-7)
- **Tables M1-M16:** Noise × Order (one per full-coverage method)
- Complete raw nRMSE values for all 1,310 experimental data points

**Best for:** Exact numerical values, verifying specific claims

### 2. 📈 Filtered Performance Plots  
**File:** `supplementary_plots.pdf` (8 pages, 1.5 MB)

**Contains:**
- **Figure 1:** Heatmap of top 10 methods across all orders
- **Figure 2:** Noise sensitivity curves for top 5 methods (8 subplots)
- **Figure 3:** Order progression at moderate noise (10^-3)
- **Figure 4:** Per-method heatmaps (noise × order) for top 6 methods

**Best for:** Visual patterns, method comparison, presentations

### 3. 📄 Main Paper
**File:** `paper.pdf` (52 pages, 587 KB)

**Contains:** Complete benchmark study with aggregate analysis

## Quick Reference

### For Your Colleagues' Questions:

**"Show me the raw data for order 4"**
→ `supplementary_data_tables_final.pdf`, Table S5

**"Which methods work at high noise?"**
→ `supplementary_plots.pdf`, Figure 2 (noise sensitivity)

**"Can I see the AAA failure?"**
→ Tables: Table M1, Plots: Figure 4 (AAA missing from top 6)

**"What about GP-Julia-AD across all conditions?"**
→ Tables: Table M4, Plots: Figure 4 subplot 1

**"Visual overview of everything"**
→ `supplementary_plots.pdf`, Figure 1 (heatmap)

## What Was Generated

### Data Tables (Python script → LaTeX → PDF)
```
generate_detailed_tables.py     → Extracts from CSV
detailed_tables_content.tex     → 627 lines of tables  
supplementary_data_tables_final.tex → Full document
supplementary_data_tables_final.pdf → 25 pages
```

### Filtered Plots (Python script → PNG → PDF)
```
generate_supplementary_plots.py → Creates 4 plots
paper_figures/supplementary/*.png → 4 PNGs (268-341 KB each)
supplementary_plots.tex → Document with explanations
supplementary_plots.pdf → 8 pages
```

## Filtering Strategy

All plots use **intelligent filtering** to show useful comparisons:

- **Heatmaps:** nRMSE capped at 2-10 depending on context
- **Line plots:** nRMSE capped at 10
- **Method ranking:** Uses filtered average (excludes catastrophic failures)

**Why?** Unfiltered data has nRMSE ranging from 10^-9 to 10^22. Without filtering:
- Log scales compress useful range (0.1-1.0) to invisibility
- Linear scales make everything except failures look identical
- Rankings dominated by methods that fail least catastrophically vs perform best

**Trade-off:** Lose magnitude information for extreme failures, gain ability to compare usable methods.

## Data Integrity

✅ **Tables:** Unfiltered raw data (all values shown exactly)  
✅ **Plots:** Filtered for visualization (extreme outliers capped)  
✅ **Source:** All from `comprehensive_summary.csv` (1,310 data points)  
✅ **Reproducible:** Scripts provided, can regenerate with different filters  

## File Sizes

```
supplementary_data_tables_final.pdf    159 KB    25 pages    Raw data
supplementary_plots.pdf              1,500 KB     8 pages    Visualizations
paper.pdf                              587 KB    52 pages    Main paper
TOTAL                                2,246 KB    85 pages
```

## Using the Venv (For Future Updates)

To regenerate or modify:

```bash
cd /home/orebas/derivative_estimation_study/report
source ../venv/bin/activate  # IMPORTANT - activates venv

# Regenerate tables
python generate_detailed_tables.py > detailed_tables_content.tex
# (reassemble into final .tex, then pdflatex)

# Regenerate plots
python generate_supplementary_plots.py
# (plots saved to paper_figures/supplementary/)

# Then rebuild PDFs
pdflatex supplementary_plots.tex
pdflatex supplementary_data_tables_final.tex
```

**Note:** Always `source ../venv/bin/activate` first to avoid numpy version conflicts!

## Navigation Aids

- **DETAILED_DATA_INDEX.md** - Quick reference for table navigation
- **SUPPLEMENTARY_MATERIALS_README.md** - Full documentation for tables
- **This file** - Overview of complete package

## Next Steps

For colleagues:
1. ✅ Share both PDFs (tables + plots)
2. ✅ Use DETAILED_DATA_INDEX.md for quick table lookup
3. ✅ Point to specific figures/tables when discussing claims

For publication:
1. ✅ Include both as supplementary materials
2. ✅ Reference in main paper: "See supplementary materials S1-S8, M1-M16"
3. ✅ Plots can be extracted for presentations (PNGs in paper_figures/supplementary/)

---
**Generated:** October 20, 2025  
**Total data points:** 1,310 (3 trials × 436 configurations, with coverage gaps)  
**Methods shown:** 27 total (16 full-coverage in Part II tables, 10 top methods in plots)
