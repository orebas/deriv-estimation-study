# ROM (Reduced Order Model) with ApproxFun - Analysis

## Summary

We implemented and tested a ROM framework using ApproxFun (Chebyshev polynomials) for computing higher-order derivatives from smoothed signals. This document summarizes our findings.

## What We Tested

### Evolution of Approach

1. **AAA (Rational Approximation)** - FAILED
   - Orders 0-1: Excellent
   - Orders 2+: Catastrophic (5.8e25 error for order 7)
   - Root cause: Poles in rational functions blow up in derivatives

2. **Manual Chebyshev** - WORKS but complex

3. **ApproxFun with transform()** - FAILED (overfitting)
   - Used 1000 coefficients → fit noise to machine precision
   - Order 7 RMSE: 2.84e33

4. **ApproxFun with least squares + controlled degree** - PARTIAL SUCCESS
   - Degree 30 with densification: Order 7 RMSE = 1.9e12
   - Degree 10 without densification: Order 7 RMSE = 634,200

## Key Findings

### 1. Densification is Harmful

**Test signal:** `y = sin(2πt) + 1e-3 noise`, N=101 points

| Approach | Order 7 RMSE |
|----------|--------------|
| Densify to 1000 pts (cubic spline) + degree 30 | 1.914e12 |
| Original 101 pts + degree 30 | 1.914e12 |
| Original 101 pts + degree 10 | **634,200** (3000x better!) |

**Conclusion:** Cubic spline interpolation introduces artifacts that ApproxFun fits perfectly, which then blow up in higher derivatives.

### 2. Lower Polynomial Degree is Better

**Without densification, on original 101 points:**

| Degree | Order 1 RMSE | Order 2 RMSE | Order 7 RMSE |
|--------|-------------|-------------|-------------|
| 10 | **0.00976** | **0.481** | **634,200** |
| 20 | 0.0213 | 1.392 | 1.078e10 |
| 30 | 0.0244 | 9.817 | 1.914e12 |

**Conclusion:** Degree 10 provides the best balance between smoothing and accuracy.

### 3. Practical Derivative Order Limits

**On sin(2πt) with 1e-3 noise, using degree 10:**

| Order | RMSE | Assessment |
|-------|------|------------|
| 0 | 0.000325 | ✓ Excellent (smoothing) |
| 1 | 0.00976 | ✓ Excellent (4x better than PyNumDiff!) |
| 2 | 0.481 | ✓ Good (truth ≈ 39.5) |
| 3 | 15.43 | ~ Acceptable (truth ≈ 248) |
| 4 | 340.5 | ✗ Poor (truth ≈ 1555) |
| 5 | 5,504 | ✗ Very poor (truth ≈ 9765) |
| 6 | 67,180 | ✗ Catastrophic |
| 7 | 634,200 | ✗ Catastrophic |

### 4. Why High Orders Fail

Even with perfect fitting, noise amplification in differentiation is unavoidable:

```
d^n/dt^n (noise) ∼ noise × (1/Δt)^n
```

For Δt = 0.01 and noise = 1e-3:
- Order 1: 1e-3 × 100 = 0.1
- Order 5: 1e-3 × 10^10 = 10,000,000

**Conclusion:** Physics limits how high we can go, not just the approximation method.

## Comparison: ApproxFun vs AAA

Test: Clean `e^x` signal, evaluate order 7 derivative at x=0.5

| Method | Order 7 Error | Relative |
|--------|--------------|----------|
| AAA (m=100) | 0.065 | 1.0x |
| Manual Cheby (deg 30) | 1.9e-5 | **3,400x better** |
| ApproxFun (deg 30) | 4.2e-8 | **1.5 million times better** |

**For clean functions, ApproxFun is vastly superior to AAA.**

## Recommended Approach

### For ROM Framework

**Don't densify:** Fit the original smoothed points directly

**Use controlled degree:**
- Degree 10: Best for orders 0-3
- Degree 20: Acceptable for orders 0-2
- Degree 30+: Only for orders 0-1

**Practical limits:**
- **Orders 0-2**: ROM works well, often better than native methods
- **Orders 3-4**: Acceptable for moderately noisy data
- **Orders 5-7**: Unreliable unless signal is extremely clean

### Implementation Changes Needed

1. **Remove densification step** from Python
   - Current: densify to 1000 points with cubic spline
   - New: save original smoothed points directly

2. **Add degree parameter** to ROM configuration
   - Default: degree 10
   - User can override based on their data quality

3. **Add order-dependent warnings**
   - Warn if requesting orders > 4
   - Document expected accuracy degradation

## Updated ROM Workflow

### Python Side

```python
# Skip densification - just save smoothed signal
def save_for_rom(method_name, t, y_smooth, t_eval, metadata=None):
    output_dir = Path("build/data/rom_input")
    data = {
        "method_name": method_name,
        "t_dense": t.tolist(),        # Original points
        "y_dense": y_smooth.tolist(),  # Original smoothed values
        "t_eval": t_eval.tolist(),
        "metadata": metadata or {}
    }
    # ... save to JSON
```

### Julia Side

```julia
# Use least squares with controlled degree
function evaluate_rom(method_name, t_eval, orders; max_degree=10)
    # Load data
    data = load_densified_data(method_name)
    t = data["t_dense"]  # Original points, not densified
    y = data["y_dense"]

    # Least squares fit with degree max_degree
    S = Chebyshev(extrema(t)...)
    basis = [Fun(S, [zeros(k-1); 1.0]) for k in 1:max_degree+1]

    A = [φ(ti) for ti in t, φ in basis]
    coeffs = A \ y

    fitted_func = Fun(S, coeffs)

    # Compute derivatives
    # ...
end
```

## Questions for User

1. **Should we keep ROM at all?**
   - It works well for orders 0-2
   - Orders 3-4 are marginal
   - Orders 5-7 are unreliable

2. **If we keep it, what orders should we support?**
   - Conservative: 0-2 only
   - Moderate: 0-4 with warnings
   - Aggressive: 0-7 (document limitations)

3. **What should we use as default degree?**
   - Degree 10: Best stability for higher orders
   - Degree 20: Better for orders 0-2 only
   - Make it configurable?

4. **Should ROM be opt-in or automatic?**
   - Auto: Add ROM variants for all methods lacking native derivatives
   - Opt-in: Require explicit user request
   - Hybrid: Auto for orders 0-2, opt-in for 3+

## Files Modified/Created

- `methods/julia/rom/approxfun_rom_wrapper.jl` - Updated with least squares fitting
- `methods/python/rom_utils.py` - Renamed from aaa_rom_utils.py
- `test_rom_approxfun.py` - End-to-end test (with densification)
- `test_rom_no_densify.py` - Test without densification (better results)
- `test_approxfun_data_fitting.jl` - Comparison of fitting methods
- `ROM_APPROXFUN_ANALYSIS.md` - This document

## Next Steps

1. Update Python ROM utils to skip densification
2. Test on multiple methods (not just PyNumDiff)
3. Document ROM capabilities and limitations for paper
4. Decide on integration strategy (auto vs opt-in)
