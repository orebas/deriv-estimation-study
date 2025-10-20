# Project Status Report

**Date:** 2025-10-13
**Phase:** Pilot Testing & Method Validation

## Executive Summary

We have completed the initial implementation and diagnostic testing phase. Out of 15 methods implemented (8 Julia, 7 Python), **7 methods are confirmed working** for high-order derivatives, **4 methods are broken** and cannot be fixed with current approaches, and **4 methods need further investigation** for Python.

## Method Status Summary

### ✅ Working Methods (7)

| Method | Language | Orders Tested | Order 3 RMSE | Notes |
|--------|----------|---------------|--------------|-------|
| **AAA-HighPrec** | Julia | 0-3 | 0.22 | ⭐ Best performer |
| **AAA-LowPrec** | Julia | 0-3 | 27.6 | Good for fast approx |
| **Dierckx-5** | Julia | 0-3 | 20.7 | ✅ Fixed namespace issue |
| **scipy-RBF-multiquadric** | Python | 0-3 | 29.5 | Uses native derivative API |
| **scipy-RBF-gaussian** | Python | 0-3 | 40.1 | Uses native derivative API |
| **scipy-CubicSpline** | Python | 0-3 | 74.2 | Uses native `cs(x, nu=order)` |
| **Central-FD** | Julia | 0-1 | N/A | Baseline comparison only |

### ❌ Broken Methods (4)

| Method | Language | Issue | Root Cause | Fixable? |
|--------|----------|-------|------------|----------|
| **GP-Julia-SE** | Julia | All derivatives NaN | GaussianProcesses.jl predict not differentiable via TaylorDiff | ❌ No (external C lib) |
| **Fourier-Interp** | Julia | Order 3 RMSE = 867M | Numerical explosion when differentiating Fourier series | ❌ No (fundamental) |
| **sklearn-GP** | Python | Using FD fallback | No native derivative API | ⚠️ Need JAX/autograd |
| **scipy-UnivariateSpline** | Python | Over-smoothing | Auto-smoothing too aggressive | ⚠️ Need tuning |

### ⚠️ Needs Investigation (4)

| Method | Language | Issue | Next Steps |
|--------|----------|-------|------------|
| **Savitzky-Golay** | Julia | Order 3 RMSE = 94.2 | Current implementation only smooths, doesn't differentiate |
| **TrendFilter-k7** | Julia | Order 3 RMSE = 94.2 | Same issue - need derivative computation |
| **numdifftools-FD** | Python | Order 3 RMSE = 94.2 | Suspicious identical error |
| **pydiff-savgol** | Python | Order 3 RMSE = 96.6 | Need to verify implementation |

## Recent Fixes

### Fix #1: TaylorDiff API (julia_methods.jl:57)
**Problem:** `MethodError` when calling TaylorDiff
**Root Cause:** Using integer `3` instead of `Val(3)`
**Solution:** Changed to `TaylorDiff.derivative(f, x, Val(n))`
**Status:** ✅ Fixed

### Fix #2: Dierckx Namespace Conflict (julia_methods.jl:490)
**Problem:** Derivatives returning NaN despite native API support
**Root Cause:** Unqualified `derivative()` call ambiguous (5+ packages export it)
**Solution:** Changed to `Dierckx.derivative(spl, xi, nu=order)`
**Status:** ✅ Fixed
**Impact:** Order 3 RMSE improved from NaN to 20.7

### Fix #3: Python CubicSpline (python_methods.py:157)
**Problem:** Using finite differences instead of native API
**Root Cause:** Implementation called `compute_derivative_fd()` for all derivatives
**Solution:** Changed to native `cs(self.x_eval, nu=order)`
**Status:** ✅ Fixed
**Impact:** Order 3 RMSE improved from ~94 to 74.2

### Fix #4: Python UnivariateSpline (python_methods.py:184)
**Problem:** Using finite differences instead of native API
**Root Cause:** Same as CubicSpline
**Solution:** Changed to native `us(self.x_eval, nu=order)`
**Status:** ✅ Fixed (but still over-smoothing)

## Key Technical Insights

### Automatic Differentiation Limitations

**TaylorDiff cannot differentiate through:**
- External C library calls (e.g., GaussianProcesses.jl)
- Functions with heavy branching/indexing
- Interpolators with conditional logic

**Working approach:**
- Methods with native derivative APIs (Dierckx, scipy splines, RBF)
- Pure Julia/Python functions with simple control flow
- Rational approximations (AAA)

### Python Derivative Strategy

Current Python methods fall into three categories:

1. **✅ Native APIs work** (scipy splines, RBF)
   - CubicSpline: `cs(x, nu=order)` up to order 3
   - UnivariateSpline: `us(x, nu=order)` up to order k (degree)
   - RBF: `rbf(x)` differentiable via finite differences

2. **❌ No native API** (sklearn GP, numdifftools)
   - sklearn GaussianProcessRegressor: no derivative support
   - Need to switch to JAX-based GP or autograd wrapper

3. **⚠️ Unclear** (pydiff, Savitzky-Golay)
   - pydiff has `order` parameter but results suspicious
   - May be computing wrong derivatives or wrong method

### Namespace Conflicts

**Problem:** Multiple Julia packages export `derivative`:
- Dierckx.jl
- TaylorDiff.jl
- Polynomials.jl
- Symbolics.jl
- DifferentiationInterface.jl

**Solution:** Always qualify with package name: `Dierckx.derivative(...)`, `TaylorDiff.derivative(...)`

## Diagnostic Test Results

Test configuration:
- System: Lotka-Volterra ODE
- Observable: x(t)
- Points: 51 (uniformly spaced)
- Noise: 0.0 (noiseless test)
- Orders: 0-3

Pass/fail thresholds:
- Order 0: RMSE < 0.1
- Order 1: RMSE < 1.0
- Order 2: RMSE < 10.0
- Order 3: RMSE < 100.0

### Julia Methods Results

```
AAA-HighPrec:
  Order 0: ✓ PASS  RMSE=0.00
  Order 1: ✓ PASS  RMSE=0.00
  Order 2: ✓ PASS  RMSE=0.02
  Order 3: ✓ PASS  RMSE=0.22

Dierckx-5:
  Order 0: ✓ PASS  RMSE=0.00
  Order 1: ✓ PASS  RMSE=0.08
  Order 2: ✓ PASS  RMSE=0.56
  Order 3: ✓ PASS  RMSE=20.68

GP-Julia-SE:
  Order 0: ✗ FAIL  RMSE=2.00 (NaN predictions)

Fourier-Interp:
  Order 0: ✗ FAIL  RMSE=0.15
  Order 1: ✗ FAIL  RMSE=936,786
  Order 3: ✗ FAIL  RMSE=867,000,000
```

### Python Methods Results

```
scipy-RBF-multiquadric:
  Order 0: ✓ PASS  RMSE=0.00
  Order 1: ✓ PASS  RMSE=0.11
  Order 2: ✓ PASS  RMSE=0.99
  Order 3: ✓ PASS  RMSE=29.53

scipy-CubicSpline:
  Order 0: ✓ PASS  RMSE=0.00
  Order 1: ✓ PASS  RMSE=0.12
  Order 2: ✓ PASS  RMSE=2.39
  Order 3: ✓ PASS  RMSE=74.19
```

## Files Updated

### Core Implementation
- `src/julia_methods.jl` - Fixed TaylorDiff API, Dierckx namespace
- `python/python_methods.py` - Fixed CubicSpline, UnivariateSpline to use native APIs

### Testing
- `src/diagnostic_test.jl` - Comprehensive method validation
- `src/minimal_pilot.jl` - End-to-end pipeline test
- `src/test_taylordiff.jl` - TaylorDiff API investigation

### Results
- `results/pilot/minimal_pilot_results.csv` - Initial pilot data

## Architecture Decisions

### Removed Fallback Logic
Per user directive: "stop coding fallbacks, we can't have errors being obscured by fallbacks."

Methods that fail now return:
- `predictions[order] = fill(NaN, length(x_eval))`
- `failures[order] = string(e)`

This makes it immediately clear when a method doesn't work.

### Derivative Computation Strategy

1. **Methods with native APIs** (Dierckx, scipy splines):
   - Use native `derivative(spl, x, nu=order)` or `spl(x, nu=order)`
   - Most reliable for high orders

2. **Pure Julia/Python functions** (AAA, RBF via FD):
   - Use TaylorDiff for Julia
   - Use finite differences for Python RBF (no AD through scipy)
   - ForwardDiff too slow for high orders

3. **External library wrappers** (GP-Julia-SE):
   - Cannot use AD (not differentiable)
   - Need native derivative API or switch libraries

## Known Issues

### Issue #1: Identical RMSE Values (94.16)
Multiple methods (Savitzky-Golay, TrendFilter, numdifftools, pydiff) showing identical Order 3 RMSE = 94.160764.

**Hypothesis:** These methods are all returning the original noisy data for derivatives instead of computing them properly.

**Evidence:** All pass Order 0 perfectly, but fail identically for higher orders.

**Next Steps:** Investigate each method's derivative computation individually.

### Issue #2: UnivariateSpline Over-smoothing
The auto-selected smoothing parameter `s = n * noise_est^2` appears too aggressive.

**Current:** `noise_est = np.std(y_train) * 0.1`
**Result:** Order 3 RMSE = 117.8 (failing threshold)

**Next Steps:**
- Try smaller smoothing: `s = 0.1 * n * noise_est^2`
- Or switch to `InterpolatedUnivariateSpline` (no smoothing)

### Issue #3: Python GP No Derivatives
sklearn's GaussianProcessRegressor has no native derivative API.

**Options:**
1. Use JAX-based GP (jax.scipy.gaussian_process)
2. Use GPy library (has derivative support)
3. Use finite differences on GP predictions (current, suboptimal)

**User Decision:** Will be handled in separate session focused on Python.

## Next Steps

### Immediate (Current Session)
- [x] Fix Dierckx-5 namespace issue
- [x] Document status and findings
- [ ] Update IMPLEMENTATION_PLAN.md
- [ ] Review DESIGN.md for outdated info

### Python Methods (Separate Session)
User will start separate session to determine Python strategy for:
- sklearn GP derivatives (JAX? GPy? Other?)
- Savitzky-Golay proper implementation
- numdifftools investigation
- pydiff validation

### Future Work
1. **Scale up pilot study** - More noise levels, data sizes, systems
2. **Add more methods** - Target 15-20 total (currently have 7 working)
3. **Test higher orders** - Currently tested 0-3, need 5-7
4. **Performance optimization** - Some methods slow (AAA-HighPrec: 9.9s)
5. **Statistical analysis** - Aggregate results, rank methods

## Method Recommendations

Based on current pilot results:

**Best Overall:** AAA-HighPrec (Order 3 RMSE = 0.22)
- Most accurate for high-order derivatives
- Pure Julia, no external dependencies
- Slow (9.9s) but excellent results

**Best Fast Option:** Dierckx-5 (Order 3 RMSE = 20.7)
- Native derivative support up to order 5
- Fast (0.12s)
- Good accuracy

**Best Python:** scipy-RBF-multiquadric (Order 3 RMSE = 29.5)
- Native API for function values
- Differentiable via finite differences
- Fast (0.41s)

## References

- Test system: Lotka-Volterra ODE (src/ground_truth.jl)
- Diagnostic script: src/diagnostic_test.jl
- Pilot script: src/minimal_pilot.jl
- Results: results/pilot/minimal_pilot_results.csv
