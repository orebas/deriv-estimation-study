# Detailed Data Tables - Quick Reference

## What You Asked For

You requested:
1. ✅ **nRMSE for method vs noise level, separately for each derivative level** → Tables S1-S8 (Part I)
2. ✅ **nRMSE for noise vs derivative level, for each plausible method** → Tables M1-M16 (Part II)

## Files to Open

### Primary: Detailed Tables PDF
**📄 `supplementary_data_tables_final.pdf`** (25 pages, 159 KB)

**Quick Navigation:**
- **Page 3**: Overview and interpretation guide
- **Pages 4-11**: Part I - Method × Noise tables (S1-S8), one per order
- **Pages 12-24**: Part II - Noise × Order tables (M1-M16), one per method

### Reference: Main Paper
**📄 `paper.pdf`** (52 pages)
- Contains aggregate analysis and key insights
- Tables 2-7 show summary statistics
- Section 7 discusses detailed findings

## Table Organization

### Part I: Method Performance at Each Order
| Table | Order | What It Shows |
|-------|-------|---------------|
| S1 | 0 | Function interpolation - easiest task |
| S2 | 1 | First derivative - still manageable |
| S3 | 2 | Second derivative - difficulty increases |
| S4 | 3 | Third derivative - AAA failure begins |
| S5 | 4 | Fourth derivative - major separation |
| S6 | 5 | Fifth derivative - few methods viable |
| S7 | 6 | Sixth derivative - extreme challenge |
| S8 | 7 | Seventh derivative - only robust methods survive |

**Each table shows:**
- Rows: All 27 methods
- Columns: 7 noise levels (1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 2e-2, 5e-2)
- Values: Mean nRMSE across 3 trials

### Part II: Complete Method Profiles
| Table | Method | Category | Why Look |
|-------|--------|----------|----------|
| M1 | AAA-HighPrec | Rational | See catastrophic failure |
| M2 | AAA-LowPrec | Rational | Comparison to high-precision version |
| M3 | Fourier-Interp | Spectral | Best spectral method |
| M4 | GP-Julia-AD | GP | **Top overall performer** |
| M5 | GP-Julia-SE | GP | Alternative GP kernel |
| M6-M16 | Various | Mixed | Full-coverage methods |

**Each table shows:**
- Rows: 7 noise levels
- Columns: 8 derivative orders (0-7)
- Values: Mean nRMSE across 3 trials

## Reading the Tables

### Color-Coded Interpretation
```
nRMSE < 0.1     → ✅ Excellent (green zone)
nRMSE 0.1-0.5   → ✓  Good (yellow zone)
nRMSE 0.5-1.0   → ~  Acceptable (orange zone)
nRMSE > 1.0     → ✗  Poor (red zone)
nRMSE > 10      → 💥 Catastrophic failure
```

### Special Symbols
- `---` = Method not tested at this config (out of scope or failed)
- Scientific notation (e.g., `1.2e+05`) = Extreme failure

## Key Findings (Visible in Tables)

### 1. AAA Catastrophic Failure (Table M1)
```
Order:    0      1      2      3       4        5        6         7
nRMSE:  9.7e-9  1.2e-6  2.6e-4  9.7e-2  5.8e+1  4.1e+4  3.0e+7   2.1e+10
        ✅     ✅     ✅     ~      ✗       💥      💥       💥
```
→ Works great at orders 0-2, dies at order 3+

### 2. GP-Julia-AD Consistency (Table M4)
```
Order:    0      1      2      3      4      5      6      7
nRMSE:  0.007  0.025  0.076  0.162  0.275  0.393  0.501  0.620
        ✅     ✅     ✅     ✓      ✓      ~      ~      ~
```
→ Gradual degradation, no catastrophic jumps

### 3. Noise Sensitivity (Any Table S{N}, read across)
At Order 4 (Table S5):
- GP-Julia-AD: 0.038 → 0.683 (modest increase)
- AAA-HighPrec: 57.9 → 3.9e+10 (catastrophic explosion)

## Use Cases

**Scenario 1: "I need 4th derivatives with 1% noise"**
→ Go to Table S5 (Order 4), find column "0.010", scan for nRMSE < 0.5

**Scenario 2: "Can method X handle all orders?"**
→ Find method X in Part II tables (M1-M16), scan all columns

**Scenario 3: "Which methods work at high noise?"**
→ Check Tables S1-S8, rightmost column (0.050)

**Scenario 4: "Show colleague the AAA failure**
→ Point to Table M1, orders 3-7

## Generating More Detail

If you need:
- **Filtered plots**: Fix matplotlib dependency, run `generate_supplementary_plots.py`
- **Different table layouts**: Modify `generate_detailed_tables.py`
- **Subset of methods**: Filter CSV before running scripts

## Data Integrity

✓ All tables auto-generated from `comprehensive_summary.csv`  
✓ No manual filtering or editing  
✓ Raw experimental results (3 trials per config)  
✓ Complete coverage shown (includes failures and gaps)  

## Files Created This Session

```
supplementary_data_tables_final.pdf          159 KB  Primary deliverable
supplementary_data_tables_final.tex          698 lines  LaTeX source
generate_detailed_tables.py                  Python script
detailed_tables_content.tex                  627 lines  Generated tables
SUPPLEMENTARY_MATERIALS_README.md            Full documentation
DETAILED_DATA_INDEX.md                       This file
```

## Next Steps

For publication:
1. ✅ Use supplementary PDF as-is for submission
2. ⏳ Generate plots once matplotlib fixed (optional)
3. ⏳ Add table references to main paper text if needed

For colleagues:
1. ✅ Share `supplementary_data_tables_final.pdf`
2. ✅ Point to specific tables (S1-S8 or M1-M16) for claims
3. ✅ Use this index for navigation

---
**Generated:** October 20, 2025  
**Data:** 1,310 experimental data points, 3 trials each  
**Tables:** 24 total (8 in Part I, 16 in Part II)
