# Savitzky-Golay Fix Documentation

## Bug Description

The original Savitzky-Golay implementation returned a nearest-neighbor interpolator (piecewise constant function) and then attempted to differentiate it using automatic differentiation. This produced near-zero derivatives for all orders ≥ 1, resulting in NRMSE ≈ 0.995-1.0 for 88% of test cases.

## Root Cause

**Original code (methods/julia/filtering/filters.jl, lines 67-70):**
```julia
return z -> begin
    idx = argmin(abs.(x .- z))
    return smoothed[idx]  # Step function!
end
```

This creates a step function. The derivative of a step function is zero everywhere (except at discontinuities where it's undefined).

## The Fix

Implemented standard Savitzky-Golay derivative filters, matching the approach used in scipy.signal.savgol_filter and other scientific computing libraries.

### Key Changes

1. **Added `savitzky_golay_coeffs(window, polyorder, deriv_order)`**
   - Computes analytical filter coefficients for any derivative order
   - Based on least-squares polynomial fitting
   - Returns weights to apply via convolution

2. **Added `apply_savitzky_golay_filter(...)`**
   - Applies filter coefficients directly to data
   - Scales by `dx^deriv_order` for proper units
   - Returns NaN at boundaries (handled by interpolation)

3. **Rewrote `evaluate_savitzky_golay(...)`**
   - Uses SG derivative filters instead of AD
   - Computes derivatives at original data points
   - Interpolates derivative values to evaluation points
   - Proper error handling for boundary cases

### Mathematical Details

For a local polynomial fit:
```
f(x) ≈ Σ c_j x^j  (over window)
```

The k-th derivative at x=0 is:
```
f^(k)(0) = k! * c_k
```

The SG filter computes filter weights that extract `c_k` from windowed data via least squares, then scales by `k!`.

## Testing

Tested on sin(x) over [0, 2π] with 101 points:

| Order | Expected | Predicted | Error | Status |
|-------|----------|-----------|-------|--------|
| 0 | 1.000000 | 0.999998 | 0.000002 | ✓ PASS |
| 1 | 0.000000 | 0.000000 | 0.000000 | ✓ PASS |
| 2 | -1.000000 | -0.999775 | 0.000225 | ✓ PASS |
| 3 | 0.000000 | 0.000000 | 0.000000 | ✓ PASS |

**Key metric:** std(predictions) ≈ 0.6-0.8 (proper variation, not near-zero)

## Impact

**Before fix:**
- Savitzky-Golay: 49/56 records with NRMSE = 0.994937 (88% failure)
- Affected all derivative orders ≥ 1
- Method unusable for high-order differentiation

**After fix:**
- Expected: NRMSE values should be reasonable (< 0.5 for low noise)
- Method should work for orders up to polyorder (default 5)
- Boundary effects handled gracefully

## Implementation Notes

1. **Uniform spacing assumption:** SG assumes approximately uniform grid spacing. Code warns if spacing varies > 5%.

2. **Boundary handling:** Filter cannot be applied at boundaries (need full window). Interior derivatives are interpolated to evaluation points.

3. **Parameter defaults:**
   - window = min(21, length(x)) (must be odd)
   - polyorder = 5
   - These support derivatives up to order 5

4. **Limitations:**
   - Cannot compute derivative order > polyorder (returns NaN with error message)
   - Requires window > polyorder
   - Less accurate for non-uniform grids

## References

- Savitzky, A., & Golay, M.J.E. (1964). Smoothing and differentiation of data by simplified least squares procedures. Analytical Chemistry, 36(8), 1627-1639.
- scipy.signal.savgol_filter documentation
- Press, W.H., et al. (2007). Numerical Recipes, 3rd Edition, Section 14.8

## Files Modified

- `methods/julia/filtering/filters.jl` - Complete rewrite of SG implementation
- `test_sg_fix.jl` - New test script for validation

## Next Steps

1. Re-run comprehensive study to regenerate all benchmark data
2. Compare before/after NRMSE values for Savitzky-Golay
3. Regenerate all tables and figures
4. Update paper text to note the corrected implementation
