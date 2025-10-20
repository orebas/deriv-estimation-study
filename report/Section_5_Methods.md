# 5. Methods Evaluated

This section describes the 24 derivative estimation methods analyzed in this study. Three additional methods were evaluated but excluded from final analysis due to implementation failures (see Section 4.7.2).

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
| TVRegDiff-Julia | Regularization | Julia | α (auto-tuned) | O(n) iterative | Partial (14/56) |
| TrendFilter-k2 | Regularization | Julia | Penalty (CV) | O(n) | Full (56/56) |
| TrendFilter-k7 | Regularization | Julia | Penalty (CV) | O(n) | Full (56/56) |
| Butterworth_Python | Other | Python | Filter order | O(n) | Partial (42/56) |
| KalmanGrad_Python | Other | Python | Process/obs noise | O(n) | Partial (42/56) |
| SVR_Python | Other | Python | C, ε (grid search) | O(n²) | Partial (42/56) |
| SpectralTaper_Python | Other | Python | Taper function | O(n log n) | Full (56/56) |
| Whittaker_m2_Python | Other | Python | Smoothing λ | O(n) | Partial (42/56) |
| ad_trig | Other | Julia | None | O(n) | Full (56/56) |
| chebyshev | Other | Julia | None | O(n log n) | Full (56/56) |
| fourier | Other | Julia | Filter frac | O(n log n) | Partial (42/56) |
| fourier_continuation | Other | Julia | Extension method | O(n log n) | Partial (42/56) |

**Coverage notes:**
- Full coverage (56/56): Tested across all 8 derivative orders × 7 noise levels
- Partial coverage: Missing high orders (typically 6-7) or restricted to orders 0-1 due to library/implementation limitations

**Excluded methods:** GP-Julia-SE (catastrophic numerical failure), TVRegDiff_Python (72× worse than Julia), SavitzkyGolay_Python (17,500× worse than Julia)

---

## 5.1 Gaussian Process Methods

Gaussian Process (GP) regression provides a principled Bayesian framework for function approximation and derivative estimation. GPs place a prior over functions and compute posterior distributions conditioned on observed data. Derivatives are obtained by differentiating the GP posterior mean.

### 5.1.1 GP-Julia-AD (Best Overall Performer)

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

**Hyperparameter optimization:** Length scale ℓ, signal variance σ²_f, and noise variance σ²_n optimized via Maximum Likelihood Estimation (MLE) using L-BFGS-B with 3 random restarts (Section 4.6).

**Implementation:** GaussianProcesses.jl with ForwardDiff.jl for automatic differentiation of kernel derivatives.

**Key advantages:**
- Optimal under Gaussian noise assumptions
- Built-in uncertainty quantification (predictive variance)
- Graceful degradation with noise
- Best overall performer across all derivative orders 0-7

**Computational complexity:** O(n³) for training (Cholesky factorization), O(n) per prediction point

**Coverage:** Full (56/56 configurations)

### 5.1.2 GP_RBF_Python

**Implementation:** scikit-learn GaussianProcessRegressor with RBF kernel

**Differences from GP-Julia-AD:** Uses numerical differentiation of kernel for derivatives rather than automatic differentiation. Hyperparameters optimized via MLE with L-BFGS-B.

**Performance:** Competitive with GP-Julia-AD but generally 1.5-2× higher nRMSE at high orders (5-7)

**Coverage:** Full (56/56 configurations)

### 5.1.3 GP_RBF_Iso_Python

**Implementation:** scikit-learn with isotropic RBF kernel (single length scale for all dimensions)

**Differences:** For 1D problems, effectively identical to GP_RBF_Python

**Coverage:** Full (56/56 configurations)

### 5.1.4 gp_rbf_mean

**Implementation:** Custom Python implementation using RBF kernel with non-zero mean function

**Key difference:** Estimates trend via mean function m(x) before applying GP to residuals

**Performance:** Mixed results; beneficial when signal has strong polynomial trend

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
   - Solve linear system for weights wᵢ minimizing ‖r(z_j) - f_j‖ over non-support points
   - Add point with maximum residual to support set
3. Terminate when max residual < tolerance (10⁻¹³)
4. Differentiate analytically: dr/dz computed via quotient rule on barycentric form

**Key implementation detail:** Uses BigFloat (256-bit) arithmetic throughout. Float64 version (AAA-LowPrec) exhibits 2-10× higher nRMSE at high orders.

**Strengths:**
- Excellent performance at orders 0-2 across all noise levels
- Deterministic (no stochastic optimization)
- Adaptive complexity (automatically selects number of support points)

**Failure modes:**
- **Catastrophic breakdown at orders ≥3:** Mean nRMSE >1.0 even at noise 10⁻⁸ (Section 7.2)
- Likely cause: Accumulated differentiation error in rational function derivatives
- Spurious poles can appear near evaluation points at high noise

**Recommended use:** Orders 0-2 only, noise ≤10⁻⁸ (see Section 9, Table 4)

**Computational complexity:** O(n²) for greedy support selection, O(n) evaluation

**Coverage:** Full (56/56 configurations)

### 5.2.2 AAA-LowPrec

**Differences:** Uses Float64 (64-bit) arithmetic instead of BigFloat

**Performance:** Consistently worse than AAA-HighPrec; precision is critical for stability

**Coverage:** Full (56/56 configurations)

---

## 5.3 Spectral Methods

Spectral methods represent functions in terms of global basis functions (Fourier, Chebyshev) and compute derivatives by differentiating the basis functions.

### 5.3.1 Fourier-Interp (FFT-Based Spectral Differentiation)

**Mathematical Formulation:**

Represent signal as Fourier series:
```
f(x) ≈ Σ c_k exp(i k ω x)
```

Derivatives via differentiation in frequency domain:
```
d^n f / dx^n = Σ (i k ω)^n c_k exp(i k ω x)
```

**Algorithm:**
1. Symmetrically extend signal to enforce periodicity (Lotka-Volterra is oscillatory but not strictly periodic; Section 4.7.5)
2. Compute FFT to obtain Fourier coefficients {c_k}
3. Multiply by (i k ω)^n for n-th derivative
4. Apply low-pass filter: retain lower 40% of frequency spectrum (filter fraction = 0.4; pre-tuned per Section 4.6)
5. Inverse FFT to obtain derivative in spatial domain

**Implementation:** FFTW.jl for fast Fourier transforms

**Strengths:**
- Extremely fast: O(n log n) via FFT
- Excellent for smooth, periodic signals
- Consistent performance across all derivative orders

**Weaknesses:**
- Assumes periodicity (introduces edge artifacts for non-periodic signals)
- Fixed pre-tuned filter fraction (Section 4.6) may not be optimal for all signals

**Performance:** Top-5 method overall; particularly strong at high orders (5-7) where rational methods fail

**Computational complexity:** O(n log n)

**Coverage:** Full (56/56 configurations)

### 5.3.2 Chebyshev Spectral (chebyshev)

**Mathematical formulation:** Uses Chebyshev polynomial basis T_n(x) = cos(n arccos(x)) on [-1,1]

**Advantages over Fourier:** No periodicity assumption; Chebyshev points cluster near endpoints, reducing Runge phenomenon

**Implementation:** FastTransforms.jl for Chebyshev-to-coefficient transforms

**Performance:** Mixed; competitive with Fourier at low orders, degrades at high orders

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

**Critical limitation:** Library/implementation only provides stencils up to 1st order. Coverage restricted to orders 0-1.

**Strengths:**
- Simple, well-understood
- Fast: O(n)
- No hyperparameters

**Weaknesses:**
- Noise amplification: O(√ε/h) for first derivative with noise level ε
- Poor performance at high noise (>10⁻³)
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
where s(x) is a cubic spline and λ is the smoothing parameter controlling bias-variance tradeoff.

**Smoothing parameter selection:** Generalized Cross-Validation (GCV) minimizes predicted mean squared error on held-out data (Section 4.6)

**Implementation:** Dierckx.jl (wrapper around FORTRAN FITPACK library)

**Strengths:**
- Automatic smoothing via GCV
- Fast: O(n) with banded linear systems
- Natural boundary conditions (d²s/dx² = 0 at endpoints)

**Weaknesses:**
- Limited to orders 0-5 (degree-6 polynomial pieces required for 5th derivative; library maximum)
- Performance degrades at high noise (>2%)

**Coverage:** Partial (42/56 configurations, orders 0-5 only)

### 5.5.2 ButterworthSpline_Python

**Implementation:** Python implementation combining Butterworth filter for smoothing with spline interpolation

**Approach:** Apply Butterworth low-pass filter, then fit spline to filtered data

**Coverage:** Partial (42/56 configurations, orders 0-5 only)

### 5.5.3 RKHS_Spline_m2_Python

**Formulation:** Reproducing Kernel Hilbert Space (RKHS) spline with order m=2 (penalizes 2nd derivative)

**Implementation:** Custom Python implementation with regularization parameter tuned via cross-validation

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

**Implementation:** Julia implementation with window size and polynomial degree selected based on derivative order

**Strengths:**
- Fast: O(n) via convolution
- No global optimization
- Works well for low noise (<10⁻³)

**Weaknesses:**
- Fixed window size can be suboptimal
- Poor performance at high noise (>2%)
- Boundary handling via window shrinking

**Coverage:** Full (56/56 configurations)

---

## 5.7 Regularization Methods

### 5.7.1 TVRegDiff-Julia (Total Variation Regularized Differentiation)

**Mathematical formulation:**

Estimates derivative u = df/dx by minimizing:
```
‖Eu - Δf/Δx‖² + α TV(u)
```
where E is a discretization operator, Δf/Δx are finite differences, and TV(u) = Σᵢ |uᵢ₊₁ - uᵢ| is the total variation promoting piecewise constant derivatives.

**Algorithm:** Iterative optimization (ADMM or similar) with automatic tuning of regularization parameter α

**Implementation:** Julia implementation with convergence tolerance 10⁻⁶, max 100 iterations

**Strengths:**
- Preserves discontinuities (edges)
- Robust to outliers

**Weaknesses:**
- Limited to orders 0-1 in current implementation
- Computationally expensive: iterative solver

**Coverage:** Partial (14/56 configurations, orders 0-1 only)

### 5.7.2 TrendFilter-k2

**Formulation:** Trend filtering with penalty order k=2 (penalizes 2nd discrete derivative)

**Implementation:** Convex optimization via quadratic programming

**Coverage:** Full (56/56 configurations)

### 5.7.3 TrendFilter-k7

**Formulation:** Trend filtering with penalty order k=7 (penalizes 7th discrete derivative)

**Rationale:** Higher penalty order allows more flexible fits

**Coverage:** Full (56/56 configurations)

---

## 5.8 Other Methods

This category includes diverse approaches that don't fit standard taxonomy.

### 5.8.1 ad_trig (Automatic Differentiation of Trigonometric Fit)

**Approach:** Fit trigonometric polynomial (Fourier series with limited terms), then differentiate analytically via automatic differentiation

**Implementation:** Julia with ForwardDiff.jl

**Coverage:** Full (56/56 configurations)

### 5.8.2 fourier & fourier_continuation

**fourier:** Variant of Fourier spectral method with different filtering strategy

**fourier_continuation:** Uses Fourier continuation technique to reduce Gibbs phenomenon at boundaries

**Coverage:** Partial (42/56 configurations, orders 0-5 only)

### 5.8.3 Butterworth_Python & Whittaker_m2_Python

**Butterworth_Python:** Butterworth low-pass filter followed by finite differences

**Whittaker_m2_Python:** Whittaker smoother (weighted penalized least squares) with 2nd-order penalty

**Coverage:** Partial (42/56 configurations, orders 0-5 only)

### 5.8.4 KalmanGrad_Python (Kalman Filter for Gradient Estimation)

**Approach:** State-space model with state = [f, f', f'', ...]ᵀ, Kalman filter for sequential estimation

**Advantage:** Can handle missing data and irregular sampling (not exploited in this benchmark)

**Implementation:** Python with process and observation noise tuned via cross-validation

**Coverage:** Partial (42/56 configurations, orders 0-5 only)

### 5.8.5 SVR_Python (Support Vector Regression)

**Approach:** Fit Support Vector Machine to data with RBF kernel, differentiate numerically

**Hyperparameters:** C (regularization) and ε (tube width) selected via grid search

**Implementation:** scikit-learn SVR

**Performance:** Generally poor (not designed for derivative estimation); included for comparison

**Coverage:** Partial (42/56 configurations, orders 0-5 only)

### 5.8.6 SpectralTaper_Python

**Approach:** Fourier spectral method with multitaper windowing to reduce spectral leakage

**Advantage:** Better frequency localization than standard Fourier

**Coverage:** Full (56/56 configurations)

---

## 5.9 Implementation Notes

**Cross-language consistency:** Where methods exist in both Julia and Python, parameters were matched to ensure fair comparison (Section 4.7.3). Persistent large discrepancies led to exclusion of the inferior implementation (GP-Julia-SE, TVRegDiff_Python, SavitzkyGolay_Python).

**Reproducibility:** All methods use fixed random seeds for any stochastic components (GP hyperparameter initialization, SVR grid search randomization). Julia methods benefit from just-in-time compilation with 1 warm-up run excluded from timing (Section 4.7.5).

**Partial coverage rationale:**
- Orders 6-7 missing for many methods: polynomial degree or library limitations (splines, filters)
- Orders 0-1 only for some: implementation or regularization design restrictions (TVRegDiff, Central-FD)

**Parameter tuning fairness:** All tunable methods received equivalent optimization effort (Section 4.6). GPs and splines use per-dataset optimization (MLE/GCV); Fourier methods use pre-tuned fixed parameters (potential advantage; Section 4.6).
