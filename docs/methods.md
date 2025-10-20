# Methods: Definitions, Assumptions, and Tunables

This document summarizes each implemented method, intended use, differentiability assumptions, and primary tunables. Where relevant, we note theoretical limits for higher-order derivatives.

## Spectral and Global Approximants

### Chebyshev (Python: `chebyshev`)
- Global polynomial fit via Chebyshev basis on `[min(t), max(t)]` with analytic derivatives.
- Tunables: degree selection (heuristic bounded by data size and cap).
- Notes: Susceptible to Runge-like artifacts on non-smooth signals; degree regularization advised.

### Fourier (Python: `fourier`)
- Trigonometric series on the sample interval assuming periodic extension.
- Tunables: number of harmonics (as a function of N).
- Notes: Works best for periodic or smoothly periodicizable signals; non-periodic edges degrade higher-order derivatives.

### Fourier Continuation (Python: `fourier_continuation`)
- Polynomial-trend-removed trigonometric LS fit. Derivatives are sum of polynomial and trigonometric components.
- Tunables: trend degree (`FC_TREND_DEG`), harmonics.
- Notes: Heuristic continuation; robustified variant for non-periodic data without formal boundary matching.

## Gaussian Process Regression

### SE (Julia: `GP-Julia-SE`, Python: `gp_rbf_mean`, `GP_RBF_*`)
- Analytic posterior mean derivatives of SE kernel using Hermite polynomials.
- Julia: hyperparameters via MLE with centering, scaling, and jitter escalation.
- Python: scikit-learn `RBF` with `ConstantKernel`+`WhiteKernel`; derivatives closed-form from posterior mean.
- Tunables: length scale and amplitude ranges; noise floor; jitter.
- Notes: Infinitely differentiable; conditioning requires care for small N.

### Matérn (Python: `GP_Matern*`) [disabled by default]
- Posterior mean differentiated via autograd on the kernel vector.
- Notes: Differentiability is `⌊ν⌋`, hence unsuitable for high-order derivatives at small ν; runtime cost high.

## Splines and Local Polynomial Methods

### Dierckx Splines (Julia: `Dierckx-5`)
- Smoothing spline (`k=5`) with native derivative evaluation.
- Tunables: smoothing `s` (derived from noise level), order `k`.
- Notes: Good baseline; high-order derivatives amplify noise unless heavily smoothed.

### Savitzky–Golay (Julia: `Savitzky-Golay`, Python: `SavitzkyGolay_Python`)
- Local polynomial filtering with derivative outputs; Python version uses quintic-spline fallback when needed.
- Tunables: window length, polyorder.
- Notes: High-order derivatives are sensitive to windowing and boundary handling.

### Butterworth / SVR / Kalman Grad (Python)
- `Butterworth_Python`: low-pass (filtfilt) then quintic-spline derivatives.
- `SVR_Python`: RBF-SVR smoothing, then spline derivatives.
- `KalmanGrad_Python`: constant-acceleration state model, RTS smoothing; higher derivatives via spline.
- Notes: Primarily legacy baselines; not designed for very high-order differentiation.

## Regularization Methods

### Trend Filtering (Julia: `TrendFilter-k7`)
- Lasso-based; penalizes high-order discrete differences.
- Tunables: penalty order (fixed at 7) and `λ`.
- Notes: Good for piecewise-polynomial trends; derivatives inferred via differences are rough for higher orders.

### TV-regularized Differentiation (Julia: `TVRegDiff-Julia`, Python: `TVRegDiff_Python`)
- Objective: minimize `0.5||A u − y||^2 + α ∫ φ(u′) dx` with `φ = |⋅|` or `(⋅)^2`.
- Julia: `NoiseRobustDifferentiation.tvdiff` (CG-based; preconditioning options).
- Python: vendored minimal wrapper consistent with stur86 signature.
- Tunables: `α` (regularization), iterations, kernel (`abs`/`square`), preconditioner (Julia), step tolerances.
- Notes: Recovering `u ≈ y′`; higher orders obtained by differentiating a spline fit to `u`. Performance depends strongly on `α`.

## Finite Differences (Julia: `Central-FD`, Python: `FiniteDiff_Python`)
- Retained as baselines; Python path uses smoothing + spline differentiation.
- Not recommended for high orders due to noise amplification.

## General Evaluation Notes
- Orders evaluated are intersected with method capability.
- All metrics exclude endpoints and mask non-finites.
- For reproducibility and speed, default pilots cap at order 3; extended studies should raise the cap selectively.
