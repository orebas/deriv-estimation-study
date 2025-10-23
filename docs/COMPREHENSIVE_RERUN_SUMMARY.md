# COMPREHENSIVE STUDY RE-RUN SUMMARY
## Date: 2025-10-23

================================================================================
## TIMING
================================================================================

**Figure Generation:** 59 seconds (62 plots)
**Comprehensive Study:** ~7-8 minutes estimated (21 configs × ~20-25s each)

================================================================================
## RESULTS OVERVIEW
================================================================================

### Previous State (Before Re-run)
- **Methods in CSV:** 14 (Julia only)
- **Missing:** All Python methods
- **Reason:** Python output files not created during original run

### Current State (After Re-run)
- **Methods in CSV:** 27 (13 Julia + 14 Python)
- **Raw results:** 4,306 rows
- **Summary stats:** 1,436 rows
- **Removed:** GP-Julia-SE (per user request)

================================================================================
## WORKING METHODS (27 total)
================================================================================

### Julia Methods (13):
1. AAA-HighPrec
2. AAA-LowPrec  
3. AAA-Adaptive-Diff2
4. AAA-Adaptive-Wavelet
5. GP-Julia-AD (GP-Julia-SE removed)
6. Fourier-Interp
7. Fourier-FFT-Adaptive
8. Dierckx-5
9. Savitzky-Golay
10. TrendFilter-k7
11. TrendFilter-k2
12. TVRegDiff-Julia
13. Central-FD

### Python Methods - Full Success (12):
1. chebyshev (8/8 orders)
2. fourier (8/8 orders)
3. fourier_continuation (8/8 orders)
4. gp_rbf_mean (8/8 orders)
5. Chebyshev-AICc (8/8 orders)
6. Fourier-GCV (8/8 orders)
7. Fourier-FFT-Adaptive (8/8 orders)
8. Fourier-Continuation-Adaptive (8/8 orders)
9. AAA-JAX-Adaptive-Wavelet (8/8 orders)
10. AAA-JAX-Adaptive-Diff2 (8/8 orders)
11. GP_RBF_Python (8/8 orders)
12. GP_RBF_Iso_Python (8/8 orders)

### Python Methods - Partial Success (3):
13. SavitzkyGolay_Python (6/8 orders - fails on 6,7)
14. KalmanGrad_Python (6/8 orders - fails on 6,7)
15. TVRegDiff_Python (7/8 orders - fails on 7) ✓ IN CSV

================================================================================
## FAILING METHODS (10 Python methods)
================================================================================

### Complete Failures (returns "Unknown" error):
1. **ad_trig** - All orders fail
2. **ad_trig_adaptive** - All orders fail  
3. **SpectralTaper_Python** - All orders fail

### Partial Failures (high-order derivatives fail):
4. **AAA-Python-Adaptive-Wavelet** - Orders 0-2 work, 3-7 fail
5. **AAA-Python-Adaptive-Diff2** - Orders 0-2 work, 3-7 fail
6. **Butterworth_Python** - Orders 0-5 work, 6-7 fail
7. **ButterworthSpline_Python** - Orders 0-5 work, 6-7 fail
8. **SVR_Python** - Orders 0-5 work, 6-7 fail
9. **RKHS_Spline_m2_Python** - Orders 0-5 work, 6-7 fail
10. **Whittaker_m2_Python** - Orders 0-5 work, 6-7 fail

**Pattern:** Many methods fail on higher derivative orders (6-7), likely due to:
- Numerical instability at high orders
- Insufficient regularization
- Missing implementation for high-order derivatives

================================================================================
## GENERATED VISUALIZATIONS
================================================================================

### Publication Figures (build/figures/publication/): 12 files
- Pareto frontier (nRMSE vs Time)
- Small multiples grid (top 7 methods)
- Qualitative comparison plots

### Supplemental Figures (build/figures/supplemental/): 62 files

**Per-Method (27 methods × 2 types = 54 plots):**
- 27 heatmaps (noise × order → nRMSE)
- 27 line plots (noise sensitivity by order)

**Per-Order (8 plots):**
- Order 0-7 method comparison plots

================================================================================
## NEXT STEPS
================================================================================

To fix the 10 failing Python methods, investigate:

1. **ad_trig / ad_trig_adaptive** - Check if implementation exists
2. **SpectralTaper_Python** - Check if implementation exists
3. **AAA-Python-Adaptive-***  - Debug why orders 3+ fail
4. **High-order failures (6-7)** - Add better regularization or limit max order

================================================================================
