# Complete Method Inventory

## Overview Stats
- **Total Methods**: ~40+ methods across Julia and Python
- **Julia Methods**: 21 methods
- **Python Methods**: 20+ methods (including PyNumDiff variants)

## Detailed Method Breakdown

### 🟢 Finite Difference Methods

| Method | Language | Package | Max Order | Status | Notes |
|--------|----------|---------|-----------|---------|-------|
| Central-FD | Julia | Custom | 1 | ✅ Works | Simple central differences, only order 0-1 |

### 🟢 Local Polynomial Methods (Savitzky-Golay)

| Method | Language | Package | Max Order | Status | Notes |
|--------|----------|---------|-----------|---------|-------|
| Savitzky-Golay-Fixed | Julia | Custom | 7 | ✅ Works | Fixed window=15, polyorder=7 |
| Savitzky-Golay-Adaptive | Julia | Custom | 7 | ✅ Works | Noise-adaptive window sizing |
| SG-Package-Fixed | Julia | SavitzkyGolayFilters.jl | 7 | ✅ Works | Package-based, fixed physical window h |
| SG-Package-Hybrid | Julia | SavitzkyGolayFilters.jl | 7 | ✅ Works | Hybrid adaptive (GPT-5 recommendation) |
| SG-Package-Adaptive | Julia | SavitzkyGolayFilters.jl | 7 | ✅ Works | Pure adaptive for comparison |

### 🟢 Spline Methods

| Method | Language | Package | Max Order | Status | Notes |
|--------|----------|---------|-----------|---------|-------|
| Dierckx-5 | Julia | Dierckx.jl | 5 | ✅ Works | Degree-5 splines, supports up to order 5 |
| GSS | Julia | SmoothingSplines.jl | 2 | ✅ Works | Generalized smoothing splines, limited to order 2 |
| PyNumDiff-Spline-Tuned | Python | PyNumDiff | 3 | ✅ Works | Tuned spline smoothing |

### 🟢 Spectral Methods

| Method | Language | Package | Max Order | Status | Notes |
|--------|----------|---------|-----------|---------|-------|
| Fourier-Interp | Julia | FFTW.jl | 7+ | ✅ Works | FFT-based interpolation |
| Fourier-FFT-Adaptive | Julia | FFTW.jl | 7+ | ✅ Works | Adaptive truncation based on noise |
| Fourier-FFT-Adaptive-Python | Python | NumPy/SciPy | 7+ | ✅ Works | Python version |
| Fourier-GCV | Python | Custom | 7+ | ✅ Works | GCV-based truncation |
| Fourier-Continuation-Adaptive | Python | Custom | 7+ | ✅ Works | Boundary extension techniques |
| Chebyshev-AICc | Python | NumPy | 7+ | ✅ Works | AICc model selection |
| fourier | Python | NumPy | 7+ | ✅ Works | Basic FFT differentiation |
| chebyshev | Python | NumPy | 7+ | ✅ Works | Chebyshev polynomial basis |
| fourier_continuation | Python | Custom | 7+ | ✅ Works | With boundary handling |

### 🟢 Gaussian Process Methods

| Method | Language | Package | Max Order | Status | Notes |
|--------|----------|---------|-----------|---------|-------|
| GP-Julia-AD | Julia | TaylorDiff.jl | 7+ | ✅ Excellent | Uses automatic differentiation, best GP method |
| GP-Julia-SE | Julia | Custom | 7 | ❌ Disabled | SE kernel with analytic derivatives (numerical issues) |
| GP-Julia-Matern-0.5 | Julia | Custom | 1 | ❌ Disabled | Only C^0 continuous |
| GP-Julia-Matern-1.5 | Julia | Custom | 2 | ❌ Disabled | Only C^1 continuous |
| GP-Julia-Matern-2.5 | Julia | Custom | 3 | ❌ Disabled | Only C^2 continuous |
| GP_RBF_Iso_Python | Python | scikit-learn | 2 | ⚠️ Limited | Isotropic RBF, limited derivative support |
| GP_RBF_Python | Python | scikit-learn | 2 | ⚠️ Limited | Standard RBF kernel |
| gp_rbf_mean | Python | Custom | 2 | ⚠️ Limited | Mean function only |

### 🔴 Rational Approximation Methods (AAA)

| Method | Language | Package | Max Order | Status | Notes |
|--------|----------|---------|-----------|---------|-------|
| AAA-HighPrec | Julia | BaryRational.jl | 7 | ❌ Disabled | Catastrophic for high derivatives |
| AAA-LowPrec | Julia | BaryRational.jl | 7 | ⚠️ Poor | Better but still unstable for order 5+ |
| AAA-Adaptive-Diff2 | Julia | BaryRational.jl | 7 | ⚠️ Poor | Threshold on 2nd derivative |
| AAA-Adaptive-Wavelet | Julia | BaryRational.jl | 7 | ⚠️ Poor | Wavelet-based noise estimation |

### 🟢 Regularization Methods

| Method | Language | Package | Max Order | Status | Notes |
|--------|----------|---------|-----------|---------|-------|
| TVRegDiff-Julia | Julia | Custom | 1 | ✅ Works | Total variation regularization, order 0-1 only |
| TVRegDiff_Python | Python | Custom | 1 | ✅ Works | Python version |
| TrendFilter-k7 | Julia | Custom | 7 | ❌ Disabled | Output is discrete, interpolation destroys smoothness |
| TrendFilter-k2 | Julia | Custom | 2 | ❌ Disabled | Same issue as k7 |

### 🟢 PyNumDiff Methods (Python)

| Method | Language | Package | Max Order | Status | Notes |
|--------|----------|---------|-----------|---------|-------|
| PyNumDiff-Butter-Auto | Python | PyNumDiff | 3 | ✅ Works | Butterworth filter with auto params |
| PyNumDiff-Gaussian-Auto | Python | PyNumDiff | 3 | ✅ Works | Gaussian kernel smoothing |
| PyNumDiff-Kalman-Auto | Python | PyNumDiff | 3 | ✅ Works | Kalman filtering approach |
| PyNumDiff-TV-Velocity | Python | PyNumDiff | 3 | ✅ Works | Total variation for velocity |
| PyNumDiff-TV-Iterative | Python | PyNumDiff | 1 | ⚠️ Limited | Iterative TV, unstable for higher orders |
| PyNumDiff-Spline-Tuned | Python | PyNumDiff | 3 | ✅ Works | Tuned spline parameters |

### 🟢 Other Methods

| Method | Language | Package | Max Order | Status | Notes |
|--------|----------|---------|-----------|---------|-------|
| KalmanGrad_Python | Python | Custom | 1 | ⚠️ Limited | Gradient-only Kalman filter |
| ad_trig | Python | JAX | 7+ | ✅ Works | Trigonometric basis with AD |
| ad_trig_adaptive | Python | JAX | 7+ | ✅ Works | Adaptive version |

## Summary by Performance

### 💎 Best Performers (Reliable for high-order derivatives)
1. **GP-Julia-AD** - Gold standard for smooth functions
2. **Fourier methods** - Excellent for periodic/smooth data
3. **Chebyshev methods** - Good for non-periodic smooth data
4. **Savitzky-Golay variants** - Robust local polynomial fitting

### ✅ Good Performers (Work well within limits)
1. **PyNumDiff methods** - Good up to order 3
2. **Dierckx-5** - Reliable spline up to order 5
3. **TVRegDiff** - Excellent for order 0-1 with noise

### ⚠️ Limited/Poor Performers
1. **AAA methods** - Rational functions unstable for derivatives
2. **GSS** - Limited to order 2
3. **GP Python methods** - Limited derivative support
4. **Central-FD** - Only order 1

### ❌ Disabled/Failed Methods
1. **GP-Julia-SE** - Numerical instability with Hermite polynomials
2. **Matérn kernels** - Limited smoothness
3. **TrendFilter** - Discrete output incompatible with derivatives
4. **AAA-HighPrec** - Catastrophic failure for derivatives

## Key Findings

1. **Package vs Custom**: Most successful methods use established packages (FFTW, Dierckx, PyNumDiff) with our custom parameter selection
2. **Max Order Support**: Only spectral methods and GP-Julia-AD reliably support order 7
3. **Noise Robustness**: Savitzky-Golay and PyNumDiff methods most robust to noise
4. **Computational Cost**: Spectral methods fastest, GP methods slowest