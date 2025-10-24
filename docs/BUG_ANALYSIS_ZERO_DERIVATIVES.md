# BUG ANALYSIS: Zero Derivative Issue in Julia Methods

## Date: 2025-10-23

================================================================================
## EXECUTIVE SUMMARY
================================================================================

Multiple Julia methods are returning **zero derivatives** for orders ≥1, causing nRMSE values of ~0.99. This bug **existed in the original code** and was copied during reorganization.

**Affected Methods:**
- Savitzky-Golay (orders 1-7 return zeros)
- TrendFilter-k7, TrendFilter-k2 (orders 2-7 return zeros)
- Fourier-FFT-Adaptive (orders 6-7 return zeros)

================================================================================
## ROOT CAUSE
================================================================================

The methods use a **step function** (nearest-neighbor interpolator) which is mathematically non-differentiable:

```julia
# Current implementation in fit_savitzky_golay (and similar methods)
return z -> begin
    idx = argmin(abs.(x .- z))
    return smoothed[idx]      # ← STEP FUNCTION!
end
```

When TaylorDiff computes derivatives:
- Order 0: Returns the smoothed value ✓
- Order 1+: Correctly computes derivative = 0 (step function has zero derivative) ✗

================================================================================
## EVIDENCE
================================================================================

### Test Case: f(x) = x²
- **Expected**: f(0.5)=0.25, f'(0.5)=1.0, f''(0.5)=2.0
- **Actual**: f(0.5)=0.25 ✓, f'(0.5)=0.0 ✗, f''(0.5)=0.0 ✗

### Comprehensive Study Results
```
Savitzky-Golay,order 0: nRMSE = 0.086 ✓
Savitzky-Golay,order 1: nRMSE = 0.9949 ✗
Savitzky-Golay,order 2: nRMSE = 0.9949 ✗
...
```

**Key observation**: nRMSE values are **identical** across:
- All failing derivative orders for a given method
- All noise levels (1e-8 to 0.05)

This confirms predictions are zero vectors (constant error independent of noise).

### Both Old and New Code Affected
- Current code: /home/orebas/derivative_estimation_study/methods/julia/filtering/filters.jl
- Old code: /home/orebas/tmp/pre-reorg-code/src/julia_methods.jl
- Both have **identical** step-function interpolators (lines 67-70 vs 689-692)
- Both produce **identical** nRMSE values in comprehensive results

================================================================================
## SOLUTION
================================================================================

Replace step-function interpolators with smooth interpolators that support differentiation.

### Option 1: Linear Interpolation (Fast, Simple)
```julia
using Interpolations

function fit_savitzky_golay(x, y; window = 11, polyorder = 5)
    window = isodd(window) ? window : window + 1
    window = min(window, length(y))

    smoothed = savitzky_golay_smooth(y, window, polyorder)

    # Create linear interpolator (differentiable)
    return LinearInterpolation(x, smoothed)
end
```

### Option 2: Cubic Spline (Smooth, Better for high orders)
```julia
using Interpolations

function fit_savitzky_golay(x, y; window = 11, polyorder = 5)
    window = isodd(window) ? window : window + 1
    window = min(window, length(y))

    smoothed = savitzky_golay_smooth(y, window, polyorder)

    # Create cubic spline (C² continuous, good for 2nd derivatives)
    return CubicSplineInterpolation(x, smoothed)
end
```

### Option 3: Use Dierckx.jl Spline (Already imported)
```julia
using Dierckx

function fit_savitzky_golay(x, y; window = 11, polyorder = 5)
    window = isodd(window) ? window : window + 1
    window = min(window, length(y))

    smoothed = savitzky_golay_smooth(y, window, polyorder)

    # Create smoothing spline
    spline = Spline1D(x, smoothed; k=3, s=0.0)
    return z -> spline(z)
end
```

================================================================================
## METHODS REQUIRING FIXES
================================================================================

1. **methods/julia/filtering/filters.jl**
   - `fit_savitzky_golay` (line 58-71)

2. **methods/julia/regularization/regularized.jl**
   - `fit_trend_filter` (similar step-function issue)

3. **methods/julia/spectral/fourier.jl**
   - Check `fit_fourier_fft_adaptive` (may have similar issue for high orders)

4. **Any other methods using argmin-based interpolation**

================================================================================
## RECOMMENDED ACTION
================================================================================

1. **Immediate**: Fix `fit_savitzky_golay` using Option 2 or 3 (smooth interpolation)
2. **Test**: Re-run debug test to verify derivatives are non-zero
3. **Audit**: Check all other Julia methods for similar step-function interpolators
4. **Re-run**: Comprehensive study to get corrected results
5. **Compare**: New results against methods that are working (AAA, GP, Dierckx)

================================================================================
## IMPACT
================================================================================

**Current state**:
- 13 Julia methods in comprehensive results
- 3-4 methods returning zeros for some derivative orders
- ~30% of Julia method results are invalid

**After fix**:
- All Julia methods should produce valid derivative estimates
- Can properly compare spectral/filtering methods vs. rational approximation
- Publication figures will be accurate

================================================================================
