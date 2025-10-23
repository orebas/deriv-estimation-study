# Method Catalog

## Python Methods (25 total)

### Gaussian Process (2 methods)
- `_gp_rbf_mean_derivative` - GP with RBF kernel
- `_gp_matern` - GP with Matern kernel (parameterized by nu)

### Splines (7 methods)
- `_chebyshev` - Chebyshev polynomial approximation
- `_chebyshev_aicc` - Chebyshev with AICc criterion
- `_quintic_spline_derivatives` - Quintic spline method
- `_rkhs_spline_m2` - RKHS-based spline
- `_butterworth_spline` - Butterworth filter + spline
- `_finite_diff_spline` - Finite difference + spline
- `_svr_spline` - Support Vector Regression spline

### Spectral (8 methods)
- `_fourier` - Fourier series method
- `_fourier_gcv` - Fourier with GCV criterion
- `_fourier_fft_adaptive` - Adaptive FFT-based Fourier
- `_fourier_continuation` - Fourier continuation method
- `_fourier_continuation_adaptive` - Adaptive Fourier continuation
- `_spectral_taper_derivative` - Spectral method with tapering
- `_ad_trig` - Trigonometric AD method
- `_ad_trig_adaptive` - Adaptive trigonometric AD

### Adaptive (5 methods)
- `_aaa_adaptive_base` - Base AAA adaptive method
- `_aaa_adaptive_wavelet` - AAA with wavelet adaptation
- `_aaa_adaptive_diff2` - AAA with diff2 adaptation
- `_aaa_jax_adaptive_wavelet` - JAX-based AAA wavelet
- `_aaa_jax_adaptive_diff2` - JAX-based AAA diff2

### Filtering (4 methods)
- `_whittaker_m2` - Whittaker smoothing
- `_savgol_method` - Savitzky-Golay filter
- `_kalman_grad` - Kalman filter gradient estimation
- `_tvregdiff_method` - Total variation regularized differentiation

### Infrastructure
- `MethodEvaluator` - Main class containing all methods
- `__init__` - Initialization with training/eval data
- `evaluate_method` - Dispatcher to individual methods
- `_safe_orders` - Utility for handling derivative orders
- `_validate_input_data` - Input validation

## Julia Methods (~15-20 methods)

### Gaussian Process (3-4 methods)
- `fit_gp_se_analytic` - GP with SE kernel (analytic derivatives)
- `fit_gp` - General GP fitting
- `fit_gp_ad` - GP with automatic differentiation
- `fit_gp_matern` - GP with Matern kernel

### Splines (2 methods)
- `fit_dierckx_spline` - Dierckx spline fitting
- `fit_finite_diff` - Finite difference based

### Spectral (2 methods)
- `fit_fourier` - Fourier series fitting
- `fourier_fft_deriv` - FFT-based Fourier derivatives

### Adaptive (1 method)
- `fit_aaa` - AAA rational approximation

### Filtering (2 methods)
- `fit_savitzky_golay` - Savitzky-Golay filter
- `fit_trend_filter` - Trend filtering

### Utilities
- `nth_deriv_at` - Nth derivative at point
- `nth_deriv_taylor` - Taylor-based nth derivative
- `compute_derivatives_at_points` - Multi-point derivative computation
- `bary_eval` - Barycentric evaluation
- Helper functions for Fourier, SavGol, etc.
- `evaluate_julia_method` - Method dispatcher
- `evaluate_all_julia_methods` - Batch evaluator

## Extraction Strategy

1. Create `common.py` with:
   - Base `MethodEvaluator` class
   - Shared utilities (`_safe_orders`, `_validate_input_data`)
   - Common imports

2. Create category-specific files:
   - `methods/python/gp/gaussian_process.py`
   - `methods/python/splines/splines.py`
   - `methods/python/spectral/spectral.py`
   - `methods/python/adaptive/aaa_methods.py`
   - `methods/python/filtering/filters.py`

3. Each category file will:
   - Import from `common.py`
   - Define category-specific class inheriting from base
   - Implement only methods for that category
   - Be <300 lines for auditability

4. Create `methods/python/__init__.py` to aggregate all methods
