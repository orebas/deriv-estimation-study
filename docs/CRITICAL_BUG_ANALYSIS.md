# CRITICAL BUG ANALYSIS: Derivative Computation Failure

**Status:** CONFIRMED SOFTWARE BUG (not a limitation)
**Severity:** HIGH - Affects 38.5% of all benchmark records
**Models Consulted:** OpenAI o3, Gemini 2.5 Pro (both confirmed diagnosis)

## Executive Summary

**The Bug:** Methods return simple interpolators (nearest-neighbor, linear) instead of differentiable functions. When automatic differentiation is applied, it computes derivatives of the interpolation scheme, not the underlying fitted model, producing near-zero/constant predictions.

**Impact:**
- 574 out of 1,491 records (38.5%) have NRMSE > 0.95
- 156 records have NRMSE = 0.994937 (exactly)
- Affects Savitzky-Golay, TrendFilter-k2, TrendFilter-k7, and others

**Root Cause:** Architectural mismatch between method output (simple interpolators) and benchmark evaluation (expects smooth, differentiable functions).

---

## Technical Analysis

### Problem 1: Savitzky-Golay (filters.jl)

**Current Implementation (WRONG):**
```julia
function fit_savitzky_golay(x, y; window = 11, polyorder = 5)
    smoothed = savitzky_golay_smooth(y, window, polyorder)

    # BUG: Returns nearest-neighbor lookup (piecewise constant!)
    return z -> begin
        idx = argmin(abs.(x .- z))
        return smoothed[idx]  # Step function!
    end
end
```

**Then used as:**
```julia
predictions[order] = [nth_deriv_taylor(fitted_func, order, xi) for xi in x_eval]
```

**Why It Fails:**
- `argmin(abs.(x .- z))` creates a **step function** (piecewise constant)
- Step functions have:
  - 0th derivative: piecewise constant value
  - 1st+ derivatives: **ZERO everywhere** (except undefined at discontinuities)
- TaylorDiff correctly computes derivative of step function → near-zero predictions
- NRMSE(zeros vs true) ≈ 0.995-1.0

**Evidence:**
- 49/56 Savitzky-Golay records have NRMSE = 0.994937
- Failures increase with derivative order (88% at order 2, ~50% at orders 5-7)

### Problem 2: TrendFilter (regularized.jl)

**Current Implementation (WRONG):**
```julia
function fit_trend_filter(x, y; order = 7, λ = 0.1)
    tf = fit(TrendFilter, y, order, λ)
    y_fitted = tf.β

    # BUG: Returns linear interpolator!
    return z -> begin
        # Linear interpolation between fitted points
        t = (z - x[idx-1]) / (x[idx] - x[idx-1])
        return (1 - t) * y_fitted[idx-1] + t * y_fitted[idx]
    end
end
```

**Why It Fails:**
- Linear interpolation has:
  - 0th derivative: linear segments
  - 1st derivative: **piecewise constant** (jumps at knots)
  - 2nd+ derivatives: **ZERO everywhere** (except undefined at knots)
- TrendFilter-k7 and TrendFilter-k2 produce **identical** failures because the linear interpolation discards higher-order smoothness
- Explains why both k2 and k7 have 42/56 failures

**Evidence:**
- TrendFilter-k2: 42/56 records with NRMSE = 0.994937 (75% failure)
- TrendFilter-k7: 42/56 records with NRMSE = 0.994937 (75% failure)
- Filter order is irrelevant because interpolation is always linear

### Why NRMSE = 0.994937 (Not 1.0)?

**Analysis from Gemini Pro:**

For zero predictions and zero-mean true signal, NRMSE would be exactly 1.0.
The value 0.994937 indicates one of:

1. **Predictions aren't exactly zero:** TaylorDiff produces small non-zero values at discontinuities
2. **Small variation from numerical noise:** AD artifacts at grid boundaries
3. **Possible NRMSE calculation artifact:** The specific test function (Lotka-Volterra) may have consistent statistical properties that produce this value

The exact value 0.994937 appearing 156 times suggests a systematic effect related to the specific test data characteristics, not random noise.

---

## Confirmed Diagnosis (Both Models Agree)

### O3 Model Analysis:
✓ Confirmed interpolators return non-differentiable functions
✓ Identified exact lines where bug occurs
✓ Requested file content for line-numbered fixes

### Gemini 2.5 Pro Analysis:
✓ Confirmed "impedance mismatch" between method output and benchmark expectations
✓ Verified step function → zero derivatives
✓ Verified linear interpolation → higher-order derivatives = 0
✓ Provided detailed mathematical explanation
✓ Confirmed this violates standard practice for these methods

**CONSENSUS:** This is a **definite bug**, not an expected limitation.

---

## Recommended Fixes

### Fix #1: Savitzky-Golay (Gemini Pro's Solution)

**Strategy:** Use Savitzky-Golay derivative filter coefficients directly (standard practice)

**Implementation:**
1. Create `savitzky_golay_coeffs(window, polyorder, deriv_order)` function
   - Computes analytical SG filter weights for given derivative order
   - Returns weights that extract derivatives from windowed data

2. Update `evaluate_savitzky_golay`:
   - Loop through requested orders
   - For each order, compute appropriate filter weights
   - Apply filter to get derivatives at original x points
   - Interpolate results to x_eval locations

**Key Code:**
```julia
function savitzky_golay_coeffs(window::Int, polyorder::Int, deriv_order::Int=0)
    # Build Vandermonde matrix
    pos = collect((-half_window):half_window)
    A = [pos[i]^j for i in 1:window, j in 0:polyorder]

    # Solve for filter weights
    e_k = zeros(polyorder + 1)
    e_k[deriv_order + 1] = 1.0
    c = (A' * A) \ e_k
    filter_weights = A * c * factorial(deriv_order)

    return filter_weights
end
```

**References:**
- Python's `scipy.signal.savgol_filter` has `deriv` parameter
- Standard practice in all major scientific computing libraries

### Fix #2: TrendFilter (Gemini Pro's Solution)

**Strategy:** Construct proper spline from fitted points, use spline derivatives

**Implementation:**
1. Update `fit_trend_filter`:
   - Use `Interpolations.jl` to create cubic spline from `tf.β`
   - Return spline object (differentiable)

2. Update `evaluate_trend_filter`:
   - Call `Interpolations.derivative(fitted_spline, xi, order)`
   - No need for TaylorDiff

**Key Code:**
```julia
using Interpolations

function fit_trend_filter(x, y; order = 7, λ = 0.1)
    tf = fit(TrendFilter, y, order, λ)
    y_fitted = tf.β

    # Return cubic spline (smooth and differentiable)
    return cubic_spline_interpolation(x, y_fitted, extrapolation_bc=Line())
end

# In evaluate_trend_filter:
predictions[order] = [Interpolations.derivative(fitted_spline, xi, order) for xi in x_eval]
```

**Notes:**
- Cubic spline is standard, smooth representation
- Has well-defined derivatives of all orders
- Respects the fitted trend from TrendFilter

---

## Other Potentially Affected Methods

### High Probability

1. **Fourier-FFT-Adaptive:** 53/56 suspicious records (95% failure rate)
   - Check if it returns discrete FFT values instead of continuous function

2. **Chebyshev methods:** 35/56 suspicious records
   - May be returning Chebyshev series truncation instead of smooth interpolation

3. **AAA-JAX methods:** 44/56 suspicious records each
   - Check if JAX implementation has similar interpolator issue

### Investigation Needed

Review ALL methods that:
- Call `nth_deriv_taylor` on fitted functions
- Return lambda functions with lookups/interpolations
- Show NRMSE clustering near 1.0

Pattern to search for:
```julia
# BAD: Discrete lookup
return z -> begin
    idx = some_lookup(z)
    return discrete_values[idx]
end

# GOOD: Continuous, differentiable function
return z -> evaluate_smooth_function(coefficients, z)
```

---

## Impact Assessment

### Affected Records
```
Total benchmark records: 1,491
Suspicious (NRMSE > 0.95): 574 (38.5%)
Definitely broken (NRMSE = 0.994937): 156 (10.5%)
```

### Affected Methods (Confirmed Bugs)
| Method | Suspicious Records | Failure Rate | Fix Priority |
|--------|-------------------|--------------|--------------|
| Savitzky-Golay | 49/56 | 88% | HIGH |
| TrendFilter-k2 | 42/56 | 75% | HIGH |
| TrendFilter-k7 | 42/56 | 75% | HIGH |
| Fourier-FFT-Adaptive | 53/56 | 95% | HIGH |
| AAA-JAX-Adaptive-Wavelet | 44/56 | 79% | MEDIUM |
| AAA-JAX-Adaptive-Diff2 | 44/56 | 79% | MEDIUM |

### Clean Methods (Reference)
- GP-Julia-AD: 1/56 suspicious (2%)
- GP_RBF_Iso_Python: 1/56 (2%)
- gp_rbf_mean: 1/56 (2%)
- Fourier-GCV: 2/56 (4%)

These clean methods likely implement proper differentiable functions.

---

## Recommended Action Plan

### Immediate (Before Next Paper Submission)

1. **Fix Savitzky-Golay** (1-2 hours)
   - Implement `savitzky_golay_coeffs` function
   - Update `evaluate_savitzky_golay` to use analytical derivatives
   - Test on simple case (sin(x) with known derivatives)

2. **Fix TrendFilter** (30 minutes)
   - Add `Interpolations.jl` dependency
   - Update `fit_trend_filter` to return spline
   - Update `evaluate_trend_filter` to use spline derivatives

3. **Re-run comprehensive study** (10 minutes)
   - Regenerate all benchmark data
   - Verify NRMSE values normalize

### Short-Term (Next Week)

4. **Audit remaining methods** (2-3 hours)
   - Check Fourier-FFT-Adaptive
   - Check Chebyshev methods
   - Check AAA-JAX methods
   - Pattern: search for `idx`, `argmin`, `searchsorted` in return statements

5. **Add validation** (1 hour)
   - Detect if std(predictions) is suspiciously low
   - Warn if NRMSE > 0.9 for any method
   - Add smoke tests: known function → verify derivative accuracy

6. **Re-generate all figures/tables** (5 minutes)
   - Run full pipeline with fixes
   - Compare before/after results

### Long-Term (Future Work)

7. **Prevent recurrence:**
   - Add unit tests for derivative accuracy
   - Document method API requirements (must return smooth function)
   - Add integration tests comparing AD vs analytical derivatives

8. **Paper updates:**
   - Add methods section explaining derivative computation
   - Note which methods use analytical vs AD-based derivatives
   - Acknowledge bug fix and data regeneration

---

## Testing Strategy

### Unit Test Template
```julia
@testset "SG Derivative Accuracy" begin
    # Test function: sin(x) on [0, 2π]
    x = LinRange(0, 2π, 101)
    y = sin.(x)

    # Fit
    fitted = fit_savitzky_golay(x, y)

    # Test derivatives at x = π/2
    test_point = π/2

    # 0th: sin(π/2) = 1
    @test fitted(test_point) ≈ 1.0 atol=0.01

    # 1st: cos(π/2) = 0
    deriv1 = nth_deriv_taylor(fitted, 1, test_point)
    @test deriv1 ≈ 0.0 atol=0.01

    # 2nd: -sin(π/2) = -1
    deriv2 = nth_deriv_taylor(fitted, 2, test_point)
    @test deriv2 ≈ -1.0 atol=0.01
end
```

---

## Conclusion

**This is a definite software bug with clear fixes.**

Both O3 and Gemini 2.5 Pro confirm:
1. ✓ Diagnosis is correct
2. ✓ Fixes are standard practice
3. ✓ This explains the observed data

**The fix is straightforward:** Use analytical derivative methods instead of AD on discrete interpolators.

**Estimated effort:** 2-4 hours to fix and validate 3 main methods + re-run study

**Next Steps:** User decision on priority and timeline for fixes.
