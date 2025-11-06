# ROM (Reduced Order Model) Exploration

This directory contains an **exploratory investigation** into using ROM frameworks for derivative estimation. The work was set aside as it did not provide improvements over existing methods.

## What Was Explored

The goal was to create a generic framework for computing higher-order derivatives (orders 2-7) for smoothing methods that don't natively support them:

1. Python: Run smoothing method, save smoothed signal
2. Julia: Build approximation, compute derivatives

### Approaches Tested

1. **AAA (Adaptive Antoulas-Anderson)** - Rational approximation
   - Result: ❌ Catastrophic failure for high-order derivatives (order 7 RMSE: 5.8e25)
   - Reason: Rational functions have poles that explode in derivatives

2. **ApproxFun with Automatic Degree** - Chebyshev polynomials
   - Result: ❌ Overfits to millions of coefficients on noisy data
   - Reason: Automatic degree selection designed for clean functions, not noisy data

3. **ApproxFun with Manual Degree (Least Squares)** - Controlled polynomial degree
   - Result: ✅ Works well with degree 10
   - Finding: 513x better than GP methods for order 7 on Lotka-Volterra data
   - BUT: Not needed for GP methods which already have TaylorDiff

## Key Findings

### RMSE Comparison on Lotka-Volterra (1% noise, 101 points)

| Method                   | Coeffs | Order 0 | Order 1 | Order 3 | Order 5 | Order 7 |
|--------------------------|--------|---------|---------|---------|---------|---------|
| GP-Julia-AD (native)     | N/A    | 0.014   | 0.82    | 89.6    | 5671    | 552,100 |
| GP → ApproxFun (auto)    | 128    | 0.014   | 0.82    | 89.6    | 5671    | 554,300 |
| **Least squares (deg 10)** | **11** | **0.66** | **2.96** | **53.5** | **510** | **1,079** |
| Least squares (deg 20)   | 21     | 0.25    | 2.08    | 520     | 57,660  | 2.77M   |

**Conclusions:**
- Least squares degree 10 is 513x better for high-order derivatives
- ApproxFun automatic degree doesn't improve GP methods
- Degree 20 overfits catastrophically

## Recommendation

**The ROM framework with least squares (degree 10) could be useful for:**
- Methods without native high-order derivative support (e.g., certain PyNumDiff methods)

**Do NOT use for:**
- GP methods (GP-Julia-AD already uses TaylorDiff optimally)
- Any method with native automatic differentiation

## Directory Contents

### Implementation Files
- `methods/julia/aaa_rom/` - AAA rational approximation (failed)
- `methods/julia/rom/approxfun_rom_wrapper.jl` - ApproxFun with least squares (works)
- `methods/julia/rational/aaa.jl` - AAA implementation
- `methods/python/rom_utils.py` - Python utilities for saving data

### Test Files
- `test_aaa_*.jl/py` - AAA exploration tests
- `test_approxfun_*.jl` - ApproxFun exploration tests
- `test_rom_*.jl/py` - ROM framework tests

### Documentation
- `ROM_APPROXFUN_ANALYSIS.md` - Detailed analysis with comparison tables

## Status

**EXPLORATION COMPLETE** - Set aside for now. The ROM framework works but:
1. GP methods don't need it (already optimal)
2. Integration with other methods requires more work
3. Benefit unclear for methods that already smooth well

Consider revisiting if we need high-order derivatives from methods lacking native support.
