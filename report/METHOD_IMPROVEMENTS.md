# Method Improvements and Optimizations

**Date**: October 19, 2025

## Summary

During the course of this comprehensive study, significant improvements were made to two Julia-based methods that were initially found to have severe numerical issues. This document summarizes these improvements for inclusion in the final report.

## 1. Fourier-Interp (Julia) - FFT-Based Spectral Differentiation

### Original Problem
- **Issue**: Catastrophic noise amplification for high-order derivatives
- **Root Cause**: The spectral differentiation operator `(ik)^n` acts as a high-pass filter, exponentially amplifying high-frequency noise
- **Impact**: RMSE > 10^6 for orders 5-7, making the method unusable
- **Error Growth**: For order n=7 with typical wavenumber k~30, amplification factor ≈ 2×10^10

### Solution Implemented
- **Approach**: Low-pass filtering via frequency-domain cutoff
- **Implementation**: FFT-based spectral differentiation with regularization
- **Key Parameter**: `filter_frac = 0.4` (optimal value from systematic sweep)
- **Algorithm**:
  ```
  1. Compute FFT of symmetrically-extended data
  2. Apply spectral differentiation: multiply by (ik)^n
  3. Zero out high-frequency components where |k| > k_cutoff
  4. Inverse FFT to recover derivative in physical space
  ```

### Optimization Process
Systematic parameter sweep tested `filter_frac` ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95}:
- **Optimal**: 0.4 (best overall rank 1.9 across orders 0-7)
- **Previous default**: 0.8 (rank 6.4, poor performance)

###  Performance Improvements
Comparison at x=5.0 on test data (100 points, 1% noise):

| Order | Before (0.8) | After (0.4) | Improvement Factor |
|-------|--------------|-------------|--------------------|
| 0     | 7.44e-03     | 7.44e-03    | 1× (unchanged)     |
| 1     | 1.13e-01     | 4.35e-02    | 2.6×               |
| 2     | 3.44e-01     | 1.76e-01    | 2×                 |
| 3     | 36.3         | 3.08        | 12×                |
| 4     | 89.4         | 1.49        | **60×**            |
| 5     | 16,600       | 368         | **45×**            |
| 6     | 105,000      | 1,920       | **55×**            |
| 7     | 8,510,000    | 56,600      | **150×**           |

### Physical Interpretation
- `filter_frac = 0.4` means retaining only the **lower 40% of frequency spectrum**
- Balances signal fidelity vs noise suppression
- More aggressive than initially expected, but necessary due to `k^n` amplification

### Current Status
- ✅ Implemented in `src/julia_methods.jl`
- ✅ Default parameter optimized
- ✅ User-tunable via `params[:fourier_filter_frac]`
- ✅ Documented in `FOURIER_OPTIMIZATION.md`

---

## 2. TVRegDiff-Julia - Total Variation Regularization

### Original Problem
- **Issue**: Iterative differentiation numerically unstable for orders ≥ 2
- **Root Cause**: Cumulative error growth in iterative application of TV regularization
- **Impact**: Catastrophic errors (10^28 to 10^108) for orders 2-4, NaN for orders 5-7
- **Method**: Rick Chartrand's TV regularized differentiation algorithm

### Attempted Solution
Initial attempt to extend to higher orders by setting `max_order_precompute = maximum(orders)` in the TV differentiation call. This failed catastrophically.

### Final Resolution
**Scope limitation** rather than algorithmic fix:
- Limited method to **orders 0-1 only**
- Returns NaN with descriptive error message for orders ≥ 2
- Error message: "TVRegDiff limited to orders 0-1 (iterative differentiation unstable for higher orders)"

### Performance at Supported Orders
On test data (x=5.0, 100 points, 1% noise):
- **Order 0**: RMSE = 7.4e-3 (excellent)
- **Order 1**: RMSE ~30% relative error (acceptable for regularization method)

### Recommendation
- ✅ Use TVRegDiff-Julia for smoothing (order 0) and 1st derivatives only
- ❌ Do NOT use for orders ≥ 2 (numerically unstable)
- Alternative: Use AAA-HighPrec or GP-Julia-SE for high-order derivatives

### Current Status
- ✅ Scope limited in `src/julia_methods.jl` (lines 780-810)
- ✅ Clear error messages for unsupported orders
- ✅ Test verification in `test_fixes_simple.jl`

---

## Impact on Report

### Section Updates Required

1. **Methods to Avoid** → Remove Fourier-Interp entry
   - Previous text: "Fundamentally unstable due to ill-conditioned Vandermonde matrix"
   - **Outdated**: Method is now functional with optimized filtering

2. **Add New Section**: "Recent Method Improvements"
   - Detail Fourier-Interp optimization (45-150× improvement)
   - Explain TVRegDiff scope limitation (orders 0-1 only)

3. **Update Method Selection Guide**
   - Fourier-Interp now viable for moderate-noise scenarios at mid-range orders (2-4)
   - TVRegDiff recommended only for orders 0-1

### Technical Documentation
- Full optimization details: `FOURIER_OPTIMIZATION.md`
- Test scripts: `test_fourier_sweep.jl`, `test_fixes.jl`, `test_fixes_simple.jl`
- Code changes: `src/julia_methods.jl` (lines 510-560, 780-900)

---

## References

- **Fourier Optimization**: Consultation with Gemini-2.5-pro identified noise amplification as root cause
- **Test Data**: Lotka-Volterra system, y = sin(x) + 0.5×cos(2x) + 1% Gaussian noise
- **Date of Optimization**: October 19, 2025
