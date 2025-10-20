# Fourier-Interp Optimization Summary

## Problem Identified
The Fourier-Interp method (FFT-based spectral differentiation) was experiencing catastrophic error growth for high-order derivatives due to **noise amplification** from the `(ik)^n` differentiation operator.

## Root Cause Analysis (via Gemini-2.5-pro consultation)
- The spectral derivative operator `(ik)^n` acts as a **high-pass filter**
- High-frequency noise components get exponentially amplified: `|noise| × k^n`
- For derivative order `n=7` with wavenumber `k~30`, amplification factor ≈ 2×10^10
- Domain extension (symmetric mirroring) was correct; problem was lack of regularization

## Solution: Low-Pass Filtering
Implemented regularization via frequency-domain filtering:

```julia
# Zero out high-frequency components above cutoff
k_cutoff = filter_frac * k_max_abs
for i in eachindex(deriv_fft)
    if abs(k[i]) <= k_cutoff
        deriv_fft[i] *= (im * k[i])^n  # Differentiate
    else
        deriv_fft[i] = 0.0  # Suppress high-frequency noise
    end
end
```

## Filter Fraction Optimization
Systematic sweep of `filter_frac` values (0.3 to 0.95) revealed:

| filter_frac | Overall Rank | Notes |
|-------------|--------------|-------|
| **0.40** | **1.9** | **Optimal balance** |
| 0.30 | 2.1 | Good for high orders, undersmooths low |
| 0.50 | 3.2 | Decent compromise |
| 0.60 | 4.5 | Starts to degrade |
| 0.70 | 4.9 | Significant degradation |
| 0.80 | 6.4 | Original default - poor |
| 0.90 | 5.8 | Severe degradation |
| 0.95 | 7.2 | Worst performance |

**Conclusion**: `filter_frac=0.4` provides best overall performance across all derivative orders.

## Error Reduction Achieved
Comparison at x=5.0 on test data (100 points, 1% noise):

| Order | Before (0.8) | After (0.4) | Improvement |
|-------|--------------|-------------|-------------|
| 0 | 7.44e-03 | 7.44e-03 | 1× (same) |
| 1 | 1.13e-01 | 4.35e-02 | **2.6×** |
| 2 | 3.44e-01 | 1.76e-01 | **2×** |
| 3 | 36.3 | 3.08 | **12×** |
| 4 | 89.4 | 1.49 | **60×** |
| 5 | 16,600 | 368 | **45×** |
| 6 | 105,000 | 1,920 | **55×** |
| 7 | 8,510,000 | 56,600 | **150×** |

## Implementation Details

### Code Changes (src/julia_methods.jl)
1. **Function signature** (line 511):
   ```julia
   function fourier_fft_deriv(ff::FourierFFT, x::Float64, n::Int;
                              filter_frac::Float64=0.4)
   ```

2. **Call site** (line 873):
   ```julia
   filter_frac = get(params, :fourier_filter_frac, 0.4)
   ```

### Parameter Tunability
Users can still override the default via params:
```julia
params = Dict(:fourier_filter_frac => 0.5)  # Custom value
```

## Physical Interpretation
- `filter_frac = 1.0`: No filtering (catastrophic noise amplification)
- `filter_frac = 0.8`: Light filtering (original, inadequate)
- `filter_frac = 0.4`: **Optimal** (balanced noise suppression vs signal preservation)
- `filter_frac = 0.3`: Aggressive filtering (over-smooths low orders)
- `filter_frac = 0.0`: Total suppression (no derivatives computed)

The optimal value of 0.4 suggests that retaining only the **lower 40% of the frequency spectrum** provides the best trade-off between:
- **Fidelity**: Preserving true signal features
- **Stability**: Suppressing noise amplification

## References
- Gemini-2.5-pro consultation provided the key insight about noise amplification
- Filter fraction optimization via systematic parameter sweep
- Test data: `sin(x) + 0.5*cos(2*x)` with 1% Gaussian noise

## Files Modified
- `src/julia_methods.jl`: Updated default filter_frac from 0.8 to 0.4
- `test_fourier_sweep.jl`: Optimization sweep script (created)
- `test_fixes.jl`: Verification test showing improved accuracy

## Date
2025-10-19
