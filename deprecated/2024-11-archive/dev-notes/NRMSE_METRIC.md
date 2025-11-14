# Normalized RMSE (nRMSE) Metric Addition

**Date**: October 19, 2025

## Motivation

While RMSE and MAE provide absolute error metrics, they don't account for the vastly different magnitudes across derivative orders. A 7th derivative can have values 10^6 times larger than the original function, making absolute errors difficult to compare across orders.

## Solution: Normalized RMSE

Following consultation with OpenAI's o3 model, we added **nRMSE** as the third primary metric:

```
nRMSE = RMSE / std(true_derivative)
```

Where:
- `RMSE` is the root mean squared error
- `std(true_derivative)` is the standard deviation of the true derivative values

## Key Properties

### 1. Order-Comparable
nRMSE = 0.15 means "error is 15% of typical signal variation" whether it's order 0 or order 7.

### 2. Zero-Crossing Robust
Divides by std of the entire derivative array, not individual points, so zero-crossings don't cause singularities.

### 3. Interpretable
- **< 0.1**: High fidelity reconstruction
- **0.1 - 0.3**: Moderate quality
- **> 0.3**: Poor performance

### 4. Captures Increased Difficulty
Noise amplification in differentiation naturally inflates nRMSE at higher orders, revealing which methods handle this challenge.

### 5. Standard Metric
Related to R²: `nRMSE² = 1 - R²` for unbiased estimators
Equivalent to coefficient of variation of RMSE

## Implementation

### Computation (Julia)
```julia
# Compute errors
rmse = sqrt(mean((pred .- true_vals).^2))
mae = mean(abs.(pred .- true_vals))

# Compute normalized RMSE
true_std = std(true_vals)
nrmse = rmse / max(true_std, 1e-12)  # Avoid division by near-zero
```

### Edge Cases
- **Constant derivatives** (std ≈ 0): Use floor of `1e-12` to prevent division by zero
- Only occurs for degenerate test cases; real derivatives from ODEs have variation

## Alternative Metrics Considered

1. **Scaled Relative Error (SRE)**: `MAE / std(true)` - similar to nRMSE but using MAE
2. **MAPE with floor**: Requires arbitrary epsilon parameter
3. **Symmetric MAPE**: Bounded but complex interpretation

nRMSE was chosen as the most widely-used, parameter-free, and interpretable option.

## Files Modified

- `src/comprehensive_study.jl`: Added nRMSE computation (lines 143-145, 188-190)
- CSV outputs now include: `rmse`, `mae`, `nrmse` columns
- Summary statistics include: `mean_nrmse`, `std_nrmse`, `min_nrmse`, `max_nrmse`

## References

- O3 consultation (October 19, 2025)
- Standard practice in signal reconstruction literature
- Related to coefficient of variation and R² metrics

## Usage in Analysis

With nRMSE, researchers can:
1. **Compare methods across orders**: "Method X maintains nRMSE < 0.2 up to order 5"
2. **Identify degradation points**: "Performance degrades (nRMSE > 0.3) at order 6 for noise > 1%"
3. **Fair comparisons**: Absolute errors favor low-order derivatives; nRMSE levels the field
