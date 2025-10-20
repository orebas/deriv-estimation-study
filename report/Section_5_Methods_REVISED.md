# 5. Methods Evaluated

This section describes the 24 derivative estimation methods analyzed in this study. Three additional methods were evaluated but excluded from final analysis due to implementation failures documented in Section 4.7.2 (GP-Julia-SE: catastrophic numerical breakdown; TVRegDiff_Python and SavitzkyGolay_Python: cross-language implementation discrepancies exceeding 50× nRMSE ratio despite parameter parity efforts).

## 5.0 Summary

**Table 1: Methods Evaluated** (24 methods from 27 candidates; 3 excluded per Section 4.7.2)

| Method | Category | Language | Key Parameter(s) | Complexity | Coverage |
|--------|----------|----------|------------------|------------|----------|
| GP-Julia-AD | Gaussian Process | Julia | Length scale (MLE) | O(n³) | Full (56/56) |
| GP_RBF_Python | Gaussian Process | Python | Length scale (MLE) | O(n³) | Full (56/56) |
| GP_RBF_Iso_Python | Gaussian Process | Python | Length scale (MLE) | O(n³) | Full (56/56) |
| gp_rbf_mean | Gaussian Process | Python | Length scale (MLE) | O(n³) | Full (56/56) |
| AAA-HighPrec | Rational | Julia | Tolerance=10⁻¹³ | O(n²) | Full (56/56) |
| AAA-LowPrec | Rational | Julia | Tolerance=10⁻¹³ | O(n²) | Full (56/56) |
| Fourier-Interp | Spectral | Julia | Filter frac=0.4 | O(n log n) | Full (56/56) |
| Dierckx-5 | Spline | Julia | Smoothing (GCV) | O(n) | Partial (42/56) |
| ButterworthSpline_Python | Spline | Python | Filter order | O(n) | Partial (42/56) |
| RKHS_Spline_m2_Python | Spline | Python | Regularization | O(n³) | Partial (42/56) |
| Central-FD | Finite Difference | Julia | None (fixed stencil) | O(n) | Partial (14/56) |
| Savitzky-Golay | Local Polynomial | Julia | Window, poly degree | O(n) | Full (56/56) |
| TVRegDiff-Julia | Regularization | Julia | α (auto-tuned) | O(n) per iter | Partial (14/56) |
| TrendFilter-k2 | Regularization | Julia | Penalty (CV) | <!-- TODO: Verify solver complexity --> | Full (56/56) |
| TrendFilter-k7 | Regularization | Julia | Penalty (CV) | <!-- TODO: Verify solver complexity --> | Full (56/56) |
| ad_trig | Spectral | Julia | None | O(n) | Full (56/56) |
| chebyshev | Spectral | Julia | None | O(n log n) | Full (56/56) |
| fourier | Spectral | Julia | Filter frac | O(n log n) | Partial (42/56) |
| fourier_continuation | Spectral | Julia | Extension method | O(n log n) | Partial (42/56) |
| SpectralTaper_Python | Spectral | Python | Taper function | O(n log n) | Full (56/56) |
| Butterworth_Python | Hybrid (Filter+FD) | Python | Filter order | O(n) | Partial (42/56) |
| KalmanGrad_Python | State-Space | Python | Process/obs noise | O(n) | Partial (42/56) |
| SVR_Python | Machine Learning | Python | C, ε (grid search) | O(n²) | Partial (42/56) |
| Whittaker_m2_Python | Regularization | Python | Smoothing λ | O(n) | Partial (42/56) |

**Coverage notes:**
- Full coverage (56/56): Tested across all 8 derivative orders × 7 noise levels
- Partial coverage: Missing high orders (typically 6-7) or restricted to orders 0-1 due to library/implementation limitations

---

## 5.1 Gaussian Process Methods

Gaussian Process (GP) regression provides a principled Bayesian framework for function approximation and derivative estimation. GPs place a prior over functions and compute posterior distributions conditioned on observed data. Derivatives are obtained by differentiating the GP posterior mean.

### 5.1.1 GP-Julia-AD

**Mathematical Formulation:**

A Gaussian Process defines a distribution over functions f ~ GP(m(x), k(x,x')) where m is the mean function (typically 0) and k is the covariance kernel. Given observations y = f(x) + ε with noise ε ~ N(0, σ²_n), the posterior predictive distribution is:

```
f(x*) | y ~ N(μ*(x*), σ²*(x*))
μ*(x*) = k*(K + σ²_n I)⁻¹ y
σ²*(x*) = k** - k*ᵀ(K + σ²_n I)⁻¹ k*
```

where K_ij = k(x_i, x_j), k* = [k(x*, x_1), ..., k(x*, x_n)]ᵀ, and k** = k(x*, x*).

**Derivative estimation:** The n-th derivative of the posterior mean is obtained by differentiating the kernel function:

```
d^n μ*(x*) / dx*^n = [d^n k(x*, x_1)/dx*^n, ..., d^n k(x*, x_n)/dx*^n] (K + σ²_n I)⁻¹ y
```

**Kernel:** Squared Exponential (SE) / RBF kernel:
```
k(x, x') = σ²_f exp(-½(x - x')² / ℓ²)
```
where σ²_f is signal variance and ℓ is length scale controlling smoothness.

**Hyperparameter optimization:** Length scale ℓ, signal variance σ²_f, and noise variance σ²_n optimized via Maximum Likelihood Estimation (MLE) using L-BFGS-B with 3 random restarts (seeded deterministically per trial; Section 4.6).

**Implementation:** GaussianProcesses.jl with ForwardDiff.jl for automatic differentiation of kernel derivatives up to order 7.

**Computational note:** For high-order derivatives (n ≥ 5), ForwardDiff uses nested dual numbers, increasing per-point evaluation cost. <!-- TODO: Verify if input/output normalization and jitter term (K + σ²I + ε_jitter·I) were used for numerical stability -->

**Built-in uncertainty:** Predictive variance σ²*(x) quantifies confidence in derivative estimates (not evaluated in this benchmark).

**Computational complexity:** O(n³) for training (Cholesky factorization of K + σ²I), O(n) per prediction point (vector-matrix products)

**Coverage:** Full (56/56 configurations)

### 5.1.2 GP_RBF_Python

**Implementation:** scikit-learn GaussianProcessRegressor with RBF kernel

**Key difference from GP-Julia-AD:** <!-- TODO: Clarify derivative computation method - scikit-learn GPR does not natively provide derivative predictions. Specify if finite-difference approximation of predictive mean was used (step size, scheme) or if kernel derivatives were manually implemented -->

**Hyperparameters:** Optimized via MLE with L-BFGS-B (same as GP-Julia-AD)

**Coverage:** Full (56/56 configurations)

### 5.1.3 GP_RBF_Iso_Python

**Implementation:** scikit-learn with isotropic RBF kernel (single length scale for all dimensions)

**Note:** For 1D problems as tested here, effectively identical to GP_RBF_Python

**Coverage:** Full (56/56 configurations)

### 5.1.4 gp_rbf_mean

**Implementation:** Custom Python implementation using RBF kernel with non-zero mean function

**Key difference:** Estimates trend via mean function m(x) before applying GP to residuals: f(x) = m(x) + GP(0, k(x,x'))

**Mean function class:** <!-- TODO: Specify mean model class (e.g., polynomial degree, basis functions) and fitting procedure -->

**Coverage:** Full (56/56 configurations)

---

## 5.2 Rational Approximation Methods

Rational approximation represents functions as ratios of polynomials: r(x) = p(x)/q(x). Unlike polynomial interpolation, rational functions can capture singularities and exhibit better convergence for smooth functions.

### 5.2.1 AAA-HighPrec (Adaptive Antoulas-Anderson Algorithm)

**Mathematical Formulation:**

The AAA algorithm constructs a rational interpolant in barycentric form:

```
r(z) = Σᵢ wᵢ fᵢ / (z - zᵢ) / Σᵢ wᵢ / (z - zᵢ)
```

where {zᵢ, fᵢ} are support points (subset of data) and {wᵢ} are weights.

**Algorithm (simplified):**
1. Initialize support set with point having maximum residual
2. Iteration k:
   - Solve least-squares problem (typically via SVD of Loewner matrix) for weights wᵢ minimizing ‖r(z_j) - f_j‖ over non-support points
   - Add point with maximum residual to support set
3. Terminate when max residual < tolerance (10⁻¹³)
4. Differentiate analytically: dr/dz computed via quotient rule on barycentric form

**Key implementation detail:** Uses BigFloat (256-bit) arithmetic throughout.

**Strengths:**
- Deterministic (no stochastic optimization)
- Adaptive complexity (automatically selects number of support points m)

**Potential failure modes:**
- Spurious poles can appear near evaluation points
- High-order rational function derivatives may accumulate error

**Computational complexity:** O(n m²) where m is number of support points (typically m ≪ n); reported as O(n²) heuristically

**Coverage:** Full (56/56 configurations)

### 5.2.2 AAA-LowPrec

**Differences:** Uses Float64 (64-bit) arithmetic instead of BigFloat

**Note:** Precision may affect numerical stability for high-order derivatives

**Coverage:** Full (56/56 configurations)

---

## 5.3 Spectral Methods

Spectral methods represent functions in terms of global basis functions (Fourier, Chebyshev, or trigonometric polynomials) and compute derivatives by differentiating the basis functions.

### 5.3.1 Fourier-Interp (FFT-Based Spectral Differentiation)

**Mathematical Formulation:**

Represent signal as Fourier series on domain [a,b] with N points:
```
f(x) ≈ Σ_{k=-N/2}^{N/2} c_k exp(i k ω x)
```
where ω = 2π / (b-a) is the fundamental frequency.

Derivatives via differentiation in frequency domain:
```
d^n f / dx^n = Σ (i k ω)^n c_k exp(i k ω x)
```

**Algorithm:**
1. Symmetrically extend signal to enforce periodicity (note: Lotka-Volterra trajectories are oscillatory but not strictly periodic; Section 4.7.5)
2. Compute FFT to obtain Fourier coefficients {c_k}
3. Multiply by (i k ω)^n for n-th derivative
4. Apply low-pass filter: retain lower 40% of frequency spectrum (filter fraction = 0.4; pre-tuned per Section 4.6)
5. Inverse FFT to obtain derivative in spatial domain

**Filtering details:** <!-- TODO: Specify passband definition (fraction of Nyquist or absolute k_max), taper/roll-off, FFT normalization convention, and extension strategy (even/odd/mirror) -->

**Implementation:** FFTW.jl for fast Fourier transforms

**Strengths:**
- Extremely fast: O(n log n) via FFT
- Suitable for smooth, periodic or near-periodic signals

**Weaknesses:**
- Periodicity assumption may introduce edge artifacts for non-periodic signals
- Fixed pre-tuned filter fraction (Section 4.6 documents potential advantage over per-dataset tuning)

**Computational complexity:** O(n log n)

**Coverage:** Full (56/56 configurations)

### 5.3.2 Chebyshev Spectral (chebyshev)

**Mathematical formulation:** Uses Chebyshev polynomial basis T_n(x) = cos(n arccos(x)) on [-1,1]

**Advantages over Fourier:** No periodicity assumption; Chebyshev points cluster near endpoints, reducing Runge phenomenon

**Derivative computation:** Via Chebyshev differentiation matrix or recurrence relations

**Implementation:** FastTransforms.jl for Chebyshev-to-coefficient transforms

**Coverage:** Full (56/56 configurations)

### 5.3.3 ad_trig (Automatic Differentiation of Trigonometric Fit)

**Approach:** Fit trigonometric polynomial (Fourier series with limited terms), then differentiate analytically via automatic differentiation

**Implementation:** Julia with ForwardDiff.jl

**Model details:** <!-- TODO: Specify number of Fourier terms, selection criterion (fixed or adaptive), and any regularization -->

**Coverage:** Full (56/56 configurations)

### 5.3.4 fourier & fourier_continuation

**fourier:** Variant of Fourier spectral method with different filtering strategy

**Filtering strategy:** <!-- TODO: Specify filter type, cutoff, and how it differs from Fourier-Interp -->

**fourier_continuation:** Uses Fourier continuation technique to reduce Gibbs phenomenon at boundaries

**Continuation method:** <!-- TODO: Specify continuation technique variant (polynomial extension, least-squares, etc.) and extension length -->

**Coverage:** Partial (42/56 configurations, orders 0-5 only)

### 5.3.5 SpectralTaper_Python

**Approach:** Fourier spectral method with multitaper windowing to reduce spectral leakage

**Multitaper details:** <!-- TODO: Specify taper type (DPSS/Slepian), time-bandwidth product NW, number of tapers K, and how derivative scaling integrates with tapering -->

**Advantage:** Better frequency localization than standard Fourier windows

**Coverage:** Full (56/56 configurations)

---

## 5.4 Finite Difference Methods

Finite difference methods approximate derivatives using linear combinations of function values on a stencil.

### 5.4.1 Central-FD (Central Finite Differences)

**Mathematical formulation:**

For evenly-spaced grid with spacing h, central difference stencils approximate derivatives:

**First derivative (3-point stencil):**
```
f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
```
Truncation error: O(h²)

**Second derivative (3-point stencil):**
```
f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²
```

**Higher orders:** Require wider stencils; n-th derivative requires ≥(n+1) points

**Implementation:** Julia implementation using standard central difference stencils

**Critical limitation:** Library/implementation provides stencils only up to 1st order. Coverage restricted to orders 0-1.

**Strengths:**
- Simple, well-understood
- Fast: O(n)
- No hyperparameters

**Weaknesses:**
- Noise amplification: For additive noise with standard deviation σ, the 3-point central difference amplifies noise to O(σ/h) in the derivative estimate
- Boundary treatment requires asymmetric stencils or extrapolation

**Coverage:** Partial (14/56 configurations, orders 0-1 only)

---

## 5.5 Spline Methods

Spline methods fit piecewise polynomials with continuity constraints at knots, then differentiate the spline.

### 5.5.1 Dierckx-5 (Smoothing Spline with GCV)

**Mathematical formulation:**

Minimizes penalized least squares:
```
Σᵢ (yᵢ - s(xᵢ))² + λ ∫ (s''(x))² dx
```
where s(x) is a spline of degree k=5 and λ is the smoothing parameter controlling bias-variance tradeoff.

**Smoothing parameter selection:** Generalized Cross-Validation (GCV) minimizes predicted mean squared error on held-out data (Section 4.6)

**Implementation:** Dierckx.jl (wrapper around FORTRAN FITPACK library)

**Derivative support:** Degree-5 splines support derivatives up to order 5 (degree-k splines provide well-defined derivatives up to order k).

**Strengths:**
- Automatic smoothing via GCV
- Fast: O(n) with banded linear systems
- Natural boundary conditions (d²s/dx² = 0 at endpoints)

**Weaknesses:**
- Limited to orders 0-5 (degree-5 spline maximum for this implementation)

**Coverage:** Partial (42/56 configurations, orders 0-5 only)

### 5.5.2 ButterworthSpline_Python

**Implementation:** Python implementation combining Butterworth filter for smoothing with spline interpolation

**Approach:** Apply Butterworth low-pass filter to data, then fit spline to filtered result

**Filter details:** <!-- TODO: Specify Butterworth filter order, cutoff frequency (fraction of Nyquist), zero-phase application (filtfilt), spline degree, and knot selection -->

**Coverage:** Partial (42/56 configurations, orders 0-5 only)

### 5.5.3 RKHS_Spline_m2_Python

**Formulation:** Reproducing Kernel Hilbert Space (RKHS) spline with order m=2 (penalizes 2nd derivative)

**Objective:** Minimizes ‖f - y‖² + λ ‖D^m f‖² where D^m is m-th derivative operator

**Implementation:** Custom Python implementation with regularization parameter tuned via cross-validation

**Kernel and solver:** <!-- TODO: Specify reproducing kernel, regularization form, solver (dense vs. low-rank), and CV strategy for λ selection -->

**Theoretical connection:** RKHS splines are equivalent to GPs with specific kernel choice

**Coverage:** Partial (42/56 configurations, orders 0-5 only)

---

## 5.6 Local Polynomial Methods

### 5.6.1 Savitzky-Golay (Local Polynomial Regression)

**Mathematical formulation:**

For each point x_i, fit a polynomial of degree d to points within a sliding window of width w via least squares:
```
p(x) = Σⱼ₌₀ᵈ aⱼ (x - xᵢ)ʲ
```

The derivative is the polynomial derivative evaluated at x_i: f^(n)(x_i) ≈ n! a_n

**Closed form:** Can be expressed as discrete convolution with fixed filter coefficients (depends on window width w, polynomial degree d, and derivative order n)

**Implementation:** Julia implementation

**Parameter mapping:** <!-- TODO: Specify window size w and polynomial degree d as functions of derivative order n and boundary handling strategy -->

**Strengths:**
- Fast: O(n) via convolution
- No global optimization

**Weaknesses:**
- Fixed window size can be suboptimal
- Boundary handling via window shrinking

**Coverage:** Full (56/56 configurations)

---

## 5.7 Regularization Methods

### 5.7.1 TVRegDiff-Julia (Total Variation Regularized Differentiation)

**Mathematical formulation:**

Estimates derivative u = df/dx by minimizing an objective combining data fidelity and total variation regularization.

**Objective:** <!-- TODO: Define precise objective function - specify operator E linking derivative to observations, data fidelity term structure (e.g., cumulative sum/integration operator A), and boundary conditions. Standard TVRegDiff minimizes (1/2)||A u - f||² + α TV(u); clarify if this variant is used or specify the exact formulation -->

**Total variation:** TV(u) = Σᵢ |uᵢ₊₁ - uᵢ| promotes piecewise constant derivatives

**Algorithm:** Iterative optimization (ADMM or similar) with automatic tuning of regularization parameter α

**Implementation:** Julia implementation with convergence tolerance 10⁻⁶, max 100 iterations

**Strengths:**
- Preserves discontinuities (edges)
- Robust to outliers

**Weaknesses:**
- Limited to orders 0-1 in current implementation
- Computationally expensive: O(n) per iteration × up to 100 iterations

**Coverage:** Partial (14/56 configurations, orders 0-1 only)

### 5.7.2 TrendFilter-k2

**Formulation:** Trend filtering with penalty order k=2 (penalizes 2nd discrete derivative)

**Objective:** Minimizes (1/2)||y - x||² + λ ||D^k x||_1 where D^k is k-th order discrete difference matrix

**Implementation:** Convex optimization via <!-- TODO: Specify solver (specialized trend filtering algorithm like PDAS, path algorithm, or generic QP/ADMM) and complexity implications -->

**Coverage:** Full (56/56 configurations)

### 5.7.3 TrendFilter-k7

**Formulation:** Trend filtering with penalty order k=7 (penalizes 7th discrete derivative)

**Rationale:** Higher penalty order allows more flexible fits

**Implementation:** Same solver as TrendFilter-k2

**Coverage:** Full (56/56 configurations)

### 5.7.4 Whittaker_m2_Python

**Formulation:** Whittaker smoother (weighted penalized least squares) with 2nd-order penalty

**Objective:** Minimizes Σᵢ wᵢ(yᵢ - zᵢ)² + λ Σⱼ (Δ^m z_j)² where Δ^m is m-th order difference and wᵢ are weights (typically 1)

**Implementation:** Python with smoothing parameter λ tuned via <!-- TODO: Specify λ selection method (CV, GCV, L-curve) -->

**Coverage:** Partial (42/56 configurations, orders 0-5 only)

---

## 5.8 Hybrid and Other Methods

### 5.8.1 Butterworth_Python (Filter-Based Differentiation)

**Approach:** Apply Butterworth low-pass filter for noise reduction, then compute derivatives via finite differences

**Filter specification:** <!-- TODO: Specify Butterworth filter order, cutoff frequency selection method, zero-phase implementation (filtfilt), and finite difference scheme post-filtering -->

**Coverage:** Partial (42/56 configurations, orders 0-5 only)

### 5.8.2 KalmanGrad_Python (Kalman Filter for Gradient Estimation)

**Approach:** State-space model with state vector [f, f', f'', ...]ᵀ, Kalman filter for sequential estimation

**State-space model:** <!-- TODO: Provide state dimension p, transition matrix F(Δt), observation matrix H, process noise covariance Q, measurement noise covariance R, initialization, and CV strategy for Q/R tuning -->

**Advantage:** Can handle missing data and irregular sampling (not exploited in this benchmark with uniform grid)

**Implementation:** Python with process and observation noise parameters tuned via cross-validation

**Coverage:** Partial (42/56 configurations, orders 0-5 only)

### 5.8.3 SVR_Python (Support Vector Regression)

**Approach:** Fit Support Vector Machine to data with RBF kernel, then differentiate the fitted function numerically

**Hyperparameters:** C (regularization strength) and ε (ε-tube width) selected via grid search

**Derivative computation:** <!-- TODO: Specify numerical differentiation scheme applied to SVR predictions (finite difference step size, scheme) -->

**Implementation:** scikit-learn SVR

**Note:** SVR is not specifically designed for derivative estimation; included for comparison with ML-based approaches

**Coverage:** Partial (42/56 configurations, orders 0-5 only)

---

## 5.9 Implementation Notes

**Cross-language consistency:** Where methods exist in both Julia and Python, parameters were matched to ensure fair comparison (Section 4.7.3). Large performance discrepancies despite parameter parity led to exclusion of the inferior implementation (see Section 4.7.2 for detailed documentation and justification).

**Reproducibility:** All methods use fixed random seeds for any stochastic components (GP hyperparameter initialization, SVR grid search randomization). Julia methods benefit from just-in-time compilation with 1 warm-up run excluded from timing (Section 4.7.5).

**Partial coverage rationale:**
- **Orders 6-7 missing for many methods:** Polynomial degree or library limitations (splines require degree ≥ order; filters/regularization limited by implementation)
- **Orders 0-1 only for some:** Implementation design restrictions (TVRegDiff, Central-FD library constraints)

**Parameter tuning fairness:** All tunable methods received equivalent optimization effort (Section 4.6). GPs and splines use per-dataset optimization (MLE/GCV); Fourier methods use pre-tuned fixed parameters, which may confer an advantage (Section 4.6 discusses this methodological concern).

**Methodological transparency:** Several method descriptions contain TODO markers indicating implementation details that should be verified from code/documentation before final publication. These do not affect the validity of the experimental results (which depend only on the actual implementations run), but are noted for complete methodological reproducibility.
