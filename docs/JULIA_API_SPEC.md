# Julia Method API Specification

This document defines the standard interface for all Julia derivative estimation methods extracted from `src/julia_methods.jl`.

## Directory Structure

```
methods/julia/
├── common.jl                    # Base types and utilities
├── gp/
│   └── gaussian_process.jl      # GP methods (5 methods)
├── rational/
│   └── aaa.jl                   # AAA rational approximation (4 methods)
├── spectral/
│   └── fourier.jl               # Fourier/FFT methods (2 methods)
├── splines/
│   └── splines.jl               # Dierckx spline (1 method)
├── filtering/
│   └── filters.jl               # Savitzky-Golay (1 method)
├── regularization/
│   └── regularized.jl           # Trend filtering + TVRegDiff (3 methods)
└── finite_diff/
    └── finite_diff.jl           # Central FD (1 method)
```

## Standard API

### Module Structure

Each category module must:
1. Import required dependencies
2. Include `common.jl` for shared utilities
3. Implement specific methods as functions
4. Export method evaluation functions

### Method Signature

All methods must conform to:

```julia
"""
    evaluate_<method_name>(
        x::Vector{Float64},
        y::Vector{Float64},
        x_eval::Vector{Float64},
        orders::Vector{Int};
        params::Dict = Dict()
    ) -> MethodResult
"""
```

### Return Type: `MethodResult`

```julia
struct MethodResult
    name::String
    category::String
    predictions::Dict{Int, Vector{Float64}}  # order => values
    failures::Dict{Int, String}              # order => error message
    timing::Float64
    success::Bool
end
```

### Common Utilities (`common.jl`)

Must provide:

```julia
# Derivative computation via AD
nth_deriv_at(f, n::Int, t)                    # ForwardDiff (recursive)
nth_deriv_taylor(f, n::Int, t)                # TaylorDiff (faster for n>1)
compute_derivatives_at_points(f, x_eval, orders; method=:taylor)

# AAA rational approximation
struct AAAApprox
    f::Vector{Float64}
    x::Vector{Float64}
    w::Vector{Float64}
end
bary_eval(z, f, x, w, tol=1e-13)
fit_aaa(x, y; tol=1e-13, mmax=100)
```

## Method Categories

### 1. Gaussian Process (`gp/gaussian_process.jl`)

**Methods**:
- `GP-Julia-SE`: Analytic SE (RBF) kernel with closed-form derivatives
- `GP-Julia-AD`: AD-based GP with generic kernels
- `GP-Julia-Matern-0.5`: Matérn-1/2 kernel (exponential)
- `GP-Julia-Matern-1.5`: Matérn-3/2 kernel (once differentiable)
- `GP-Julia-Matern-2.5`: Matérn-5/2 kernel (twice differentiable)

**Key functions**:
```julia
fit_gp_se_analytic(x, y)        # Returns: (x, order) -> derivative
fit_gp_ad(x, y)                  # Returns: function
fit_gp_matern(x, y; nu=1.5)     # Returns: function
```

### 2. AAA Rational (`rational/aaa.jl`)

**Methods**:
- `AAA-HighPrec`: Fixed tolerance (1e-14)
- `AAA-LowPrec`: Fixed or adaptive tolerance (0.1 or noise-based)
- `AAA-Adaptive-Diff2`: Diff2 noise estimation
- `AAA-Adaptive-Wavelet`: Wavelet noise estimation

**Key functions**:
```julia
fit_aaa(x, y; tol=1e-13, mmax=100)  # Returns: AAAApprox
```

### 3. Spectral (`spectral/fourier.jl`)

**Methods**:
- `Fourier-Interp`: FFT-based with fixed filtering
- `Fourier-FFT-Adaptive`: FFT-based with adaptive filtering

**Key functions**:
```julia
fit_fourier(x, y; ridge_lambda=1e-8)                    # Returns: FourierFFT
fourier_fft_deriv(ff, x, n; filter_frac=0.4)            # Returns: Float64
```

### 4. Splines (`splines/splines.jl`)

**Methods**:
- `Dierckx-5`: Quintic spline with native derivative support

**Key functions**:
```julia
fit_dierckx_spline(x, y; k=5, s=nothing, noise_level=0.0)  # Returns: (func, spl)
```

### 5. Filtering (`filtering/filters.jl`)

**Methods**:
- `Savitzky-Golay`: Local polynomial smoothing

**Key functions**:
```julia
fit_savitzky_golay(x, y; window=11, polyorder=5)        # Returns: function
savitzky_golay_smooth(y, window, polyorder)             # Returns: Vector
```

### 6. Regularization (`regularization/regularized.jl`)

**Methods**:
- `TrendFilter-k7`: Trend filtering with order-7 penalty
- `TrendFilter-k2`: Trend filtering with order-2 penalty
- `TVRegDiff-Julia`: Total variation regularized differentiation (orders 0-1 only)

**Key functions**:
```julia
fit_trend_filter(x, y; order=7, λ=0.1)                  # Returns: function
```

### 7. Finite Difference (`finite_diff/finite_diff.jl`)

**Methods**:
- `Central-FD`: Central finite difference with irregular grids

**Key functions**:
```julia
fit_finite_diff(x, y)                                    # Returns: (x, order) -> derivative
```

## Environment Variables

Methods may use environment variables for hyperparameters:
- `USE_ADAPTIVE_AAA`: Enable adaptive tolerance for AAA-LowPrec
- `FOURIER_FILTER_FRAC`: Filter fraction for Fourier-Interp
- `TF_LAMBDA`: Regularization parameter for trend filtering
- `TV_ALPHA`, `TV_ITERS`: Parameters for TVRegDiff

## Error Handling

Methods must:
1. Catch all exceptions during fitting and evaluation
2. Return `MethodResult` with populated `failures` dict on error
3. Use `NaN` for failed predictions
4. Include descriptive error messages in `failures`

## Testing

Each extracted method must:
1. Pass numerical validation against original implementation
2. Use tolerance: `rtol=1e-6`, `atol=1e-5`
3. Test all supported derivative orders
4. Verify identical behavior for standard test cases

## Dependencies

Required Julia packages:
- BaryRational (AAA)
- GaussianProcesses (GP)
- ForwardDiff, TaylorDiff (AD)
- Dierckx (splines)
- FFTW (spectral)
- Lasso (trend filtering)
- NoiseRobustDifferentiation (TVRegDiff)
- Optim, LineSearches (optimization)

## Migration Checklist

For each method:
- [ ] Extract method to category module
- [ ] Preserve all hyperparameter controls
- [ ] Maintain environment variable support
- [ ] Test against original implementation
- [ ] Document any limitations (e.g., TVRegDiff orders 0-1 only)
- [ ] Update validation tests
