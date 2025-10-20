# Bug Fixes Summary

This document summarizes all critical bug fixes applied to the derivative estimation study codebase.

## Phase 1: Critical Bug Fixes

### 1.1 TVRegDiff-Julia dx Parameter Bug ✓
**File:** `src/julia_methods.jl:669`
**Issue:** Grid spacing `dt` was computed but never passed to the `tvdiff` function, causing it to use default dx=1.0 instead of the actual spacing.
**Fix:** Changed from:
```julia
current_data = NoiseRobustDifferentiation.tvdiff(current_data, iters, alpha_scaled; ...)
```
To:
```julia
current_data = NoiseRobustDifferentiation.tvdiff(current_data, iters, alpha_scaled; dx = dt, ...)
```
**Impact:** TVRegDiff-Julia now uses correct grid spacing for derivative estimation.

### 1.2 GP Hermite Overflow Protection ✓
**File:** `src/julia_methods.jl:232-280`
**Issue:** Hermite polynomial recursion produced overflow for large input values, causing NaN derivatives for n≥2.
**Fixes Applied:**
1. Input clamping: `u_safe = clamp.(u, T(-50), T(50))` to prevent overflow in exp(-0.5*u²)
2. Scaling factor clamping: `scale = clamp(scale_raw, -1e10, 1e10)` to prevent coefficient overflow
3. Finite result validation: Return NaN if result is non-finite
4. Warning for unstable conditions when |u| > 10

**Impact:** GP-Julia-SE now produces stable derivatives up to high orders.

### 1.3 Logging Added to _clean_predictions ✓
**File:** `python/python_methods.py:854-859`
**Issue:** Silent failures when predictions contained non-finite values.
**Fix:** Added warning messages:
```python
if not np.all(np.isfinite(preds)):
    logger.warning(f"Non-finite values in {method_name} order {order}, data excluded")
```
**Impact:** Easier debugging of method failures.

### 1.4 tvregdiff.py Docstring Fix ✓
**File:** `python/python_methods.py:127`
**Issue:** Function name in docstring was incorrect.
**Fix:** Changed docstring from "tv_deriv" to "tvregdiff"
**Impact:** Documentation consistency.

---

## Phase 2: Short-Term Improvements

### 2.1 Analytic Fourier Derivatives ✓
**File:** `src/julia_methods.jl:356-399, 758-771`
**Issue:** Using TaylorDiff for Fourier series derivatives was numerically unstable.
**Fix:** Implemented closed-form trigonometric derivative formulas in `fourier_deriv`:
```julia
# d/dx[cos(kz)] = -k·sin(kz)·m
# d/dx[sin(kz)] = k·cos(kz)·m
# Pattern cycles every 4 derivatives (n % 4)
```
**Known Limitation:** Fourier interpolation via Vandermonde matrix is ill-conditioned for non-periodic data, even with analytic derivatives. Ridge regularization helps but doesn't fully resolve the issue. Consider using continuation methods instead.

**Impact:** Reduced numerical errors from AD, though underlying interpolation instability remains.

### 2.2 Chebyshev Degree Cap ✓
**File:** `python/python_methods.py:207`
**Issue:** High-degree Chebyshev polynomials (degree 30) caused catastrophic numerical instability (RMSE ~3735 for order 3).
**Fix:** Capped maximum degree at 20:
```python
deg = max(3, min(20, n_train - 1))  # Was: min(30, n_train - 1)
```
**Impact:** RMSE improved from 3735 to 121.8 for order 3 (31x reduction).

### 2.3 JSON Input Validation ✓
**File:** `python/python_methods.py:861-914`
**Issue:** No validation of input data, causing cryptic errors on malformed input.
**Fix:** Added comprehensive `_validate_input_data` function checking:
- Required fields present (`times`/`t`, `y_noisy`/`y`)
- Numeric array types
- Non-empty arrays with matching lengths
- All values finite (no NaN/Inf)
- Minimum 3 data points
- Strictly increasing time values
- Valid derivative orders (0-10)

**Impact:** Clear error messages for invalid input instead of cryptic failures.

### 2.4 Subprocess Timeouts ✓
**Files:** `src/minimal_pilot.jl:85-112`, `src/diagnostic_test.jl:99-127`, `src/pilot_study.jl:102-125`
**Issue:** Python subprocess could hang indefinitely if a method failed to complete.
**Fix:** Added timeout logic using Julia's `@async` and `timedwait`:
```julia
python_task = @async try run(cmd); :success catch e; (:error, e) end
result = timedwait(() -> istaskdone(python_task), timeout_sec; pollint=0.5)
if result == :timed_out
    @warn "Python script timed out after $(timeout_sec) seconds"
end
```
Default timeout: 300 seconds (configurable via `PYTHON_TIMEOUT` env variable)

**Impact:** Prevents indefinite hangs, allows batch processing to continue on timeout.

### 2.5 Fourier Ridge Regularization ✓
**File:** `src/julia_methods.jl:406-441`
**Issue:** Fourier interpolation matrix solve produced catastrophically large coefficients (10^13-10^14) due to ill-conditioning for non-periodic data.
**Fix:** Applied L2 ridge regularization:
```julia
AtA = A' * A
Aty = A' * y
ridge_matrix = ridge_lambda * I(N)  # Default: 1e-8
coeffs = (AtA + ridge_matrix) \ Aty
```
**Impact:** Coefficient magnitudes reduced by 12 orders of magnitude (10^14 → 10^2). Partial stabilization but fundamental non-periodicity issue remains.

---

## Test Results Summary

**Before fixes:**
- Chebyshev order 3: RMSE = 3735
- Fourier-Interp order 3: RMSE = 14.7 million (catastrophic)
- TVRegDiff-Julia: Using wrong grid spacing
- GP-Julia-SE: NaN for n≥2 derivatives

**After fixes:**
- Chebyshev order 3: RMSE = 121.8 (31x improvement)
- Fourier-Interp: Still unstable (~14.7M), marked as theoretically unsound for non-periodic data
- TVRegDiff-Julia: Correct grid spacing
- GP-Julia-SE: Stable up to high orders (RMSE = 96.1 for order 3)

---

## Known Limitations

1. **Fourier-Interp**: Vandermonde matrix approach is fundamentally ill-conditioned for non-periodic data. Ridge regularization provides partial stability but cannot fully resolve the issue. Recommend using Python's `fourier_continuation` method or FFT-based approaches instead.

2. **GP Hermite Polynomials**: While stabilized with clamping, derivatives may still be unreliable for extreme input values (|u| > 50).

3. **Subprocess Timeout**: Julia doesn't provide direct process killing, so timed-out processes may continue running in background (best-effort termination).

---

## Files Modified

### Julia
- `src/julia_methods.jl`: TVRegDiff dx, GP Hermite stability, Fourier derivatives, ridge regularization
- `src/minimal_pilot.jl`: Subprocess timeout
- `src/diagnostic_test.jl`: Subprocess timeout
- `src/pilot_study.jl`: Subprocess timeout

### Python
- `python/python_methods.py`: Chebyshev degree cap, JSON validation, logging, docstring fix

---

## Recommendations for Future Work

1. Replace Fourier-Interp with continuation-based method (e.g., FFT with periodic extension)
2. Add comprehensive unit tests for each method
3. Document hyperparameter sensitivity for TVRegDiff, GP methods
4. Add regression tests comparing against AAA-HighPrec baseline
5. Consider removing or deprecating Fourier-Interp due to fundamental instability
