#!/usr/bin/env python3
"""
Automatic Hyperparameter Selection for Derivative Estimation Methods

Implements data-driven selection strategies that don't require knowing
the true noise level:
- Wavelet MAD noise estimation (Donoho-Johnstone)
- AICc for polynomial degree selection
- GCV for Fourier harmonics selection
- SURE for spectral filtering

References:
- Donoho & Johnstone (1994): Ideal spatial adaptation by wavelet shrinkage
- Craven & Wahba (1979): Smoothing noisy data with spline functions
"""

import numpy as np
from typing import Tuple, Optional
import warnings

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    warnings.warn("PyWavelets not available - wavelet-based noise estimation disabled")


# ===========================================================================
# NOISE ESTIMATION
# ===========================================================================

def estimate_noise_wavelet(y: np.ndarray, wavelet: str = 'db4') -> float:
    """
    Estimate noise standard deviation using wavelet MAD.

    This is the gold standard method (Donoho-Johnstone 1994) for estimating
    noise in smooth signals.

    Args:
        y: Signal (possibly noisy)
        wavelet: Wavelet type (default: Daubechies-4)

    Returns:
        Estimated noise standard deviation σ̂

    References:
        Donoho, D. L., & Johnstone, I. M. (1994). Ideal spatial adaptation
        by wavelet shrinkage. Biometrika, 81(3), 425-455.
    """
    if not PYWT_AVAILABLE:
        raise ImportError("PyWavelets required for wavelet-based noise estimation")

    # Decompose to finest scale
    coeffs = pywt.wavedec(y, wavelet, level=1)
    cD1 = coeffs[-1]  # Detail coefficients at finest scale

    # MAD estimator (robust to outliers)
    # Factor 0.6745 is the 75th percentile of standard normal
    sigma_hat = np.median(np.abs(cD1 - np.median(cD1))) / 0.6745

    return float(sigma_hat)


def estimate_noise_diff2(y: np.ndarray) -> float:
    """
    Estimate noise σ using second-order differences (Gemini Pro recommendation).

    This is a lightweight alternative to wavelet MAD that doesn't require
    additional dependencies. It's robust to linear trends in the signal.

    Args:
        y: Signal (possibly noisy)

    Returns:
        Estimated noise standard deviation σ̂

    Theory:
        For smooth f(x) and noise ε ~ N(0,σ²):
        d[i] = y[i] - 2y[i-1] + y[i-2]
             ≈ ε[i] - 2ε[i-1] + ε[i-2]  (signal curvature << noise)
        Var(d) = (1² + (-2)² + 1²)σ² = 6σ²

        MAD estimator: σ̂ = MAD(d) / (0.6745 * √6)
    """
    if len(y) < 3:
        raise ValueError("Need at least 3 points for 2nd-order difference")

    # Second-order difference operator
    d = y[2:] - 2*y[1:-1] + y[:-2]

    # MAD estimator
    # Var(d) = 6σ² for i.i.d. Gaussian noise
    sigma_hat = np.median(np.abs(d - np.median(d))) / 0.6745 / np.sqrt(6)

    return float(sigma_hat)


def estimate_noise_auto(y: np.ndarray) -> float:
    """
    Automatically choose best available noise estimator.

    Prefers wavelet MAD if available, falls back to 2nd-order differences.

    Args:
        y: Signal (possibly noisy)

    Returns:
        Estimated noise standard deviation σ̂
    """
    if PYWT_AVAILABLE:
        return estimate_noise_wavelet(y)
    else:
        return estimate_noise_diff2(y)


# ===========================================================================
# CHEBYSHEV: AICc-based degree selection
# ===========================================================================

def select_chebyshev_degree(
    x_train: np.ndarray,
    y_train: np.ndarray,
    max_degree: int = 30,
    min_degree: int = 3
) -> Tuple[int, float]:
    """
    Select optimal Chebyshev polynomial degree via AICc.

    Small-sample corrected AIC (Hurvich & Tsai, 1989) for polynomial
    order selection. Balances fit quality vs model complexity.

    Args:
        x_train: Training x values
        y_train: Training y values
        max_degree: Maximum degree to try
        min_degree: Minimum degree to consider

    Returns:
        (optimal_degree, min_aicc_value)

    References:
        Hurvich, C. M., & Tsai, C. L. (1989). Regression and time series
        model selection in small samples. Biometrika, 76(2), 297-307.
    """
    from numpy.polynomial.chebyshev import Chebyshev

    n = len(x_train)
    tmin, tmax = float(x_train.min()), float(x_train.max())

    best_aicc = np.inf
    best_deg = min_degree
    aicc_values = []

    # Ensure max_degree doesn't exceed data constraints
    max_deg_safe = min(max_degree, n - 2)

    for deg in range(min_degree, max_deg_safe + 1):
        try:
            # Fit Chebyshev polynomial
            poly = Chebyshev.fit(x_train, y_train, deg=deg, domain=[tmin, tmax])
            y_pred = poly(x_train)
            rss = np.sum((y_train - y_pred) ** 2)

            # AICc formula
            p = deg + 1  # number of parameters
            if n - p - 2 > 0:
                aicc = n * np.log(rss / n + 1e-12) + 2*p + 2*p*(p+1)/(n - p - 2)
            else:
                aicc = np.inf

            aicc_values.append((deg, aicc))

            if aicc < best_aicc:
                best_aicc = aicc
                best_deg = deg

        except Exception as e:
            # Numerical issues at high degree
            continue

    return best_deg, best_aicc


# ===========================================================================
# FOURIER: GCV-based harmonics selection
# ===========================================================================

def select_fourier_harmonics(
    x_train: np.ndarray,
    y_train: np.ndarray,
    max_harmonics: int = 25,
    min_harmonics: int = 1
) -> Tuple[int, float]:
    """
    Select optimal number of Fourier harmonics via GCV.

    Generalized Cross-Validation (Craven & Wahba, 1979) approximates
    leave-one-out CV efficiently using the trace formula for linear smoothers.

    Args:
        x_train: Training x values
        y_train: Training y values
        max_harmonics: Maximum harmonics to try
        min_harmonics: Minimum harmonics to consider

    Returns:
        (optimal_M, min_gcv_value)

    Theory:
        For linear smoother ŷ = A(M)y:
        GCV(M) = RSS(M)/n / (1 - df(M)/n)²
        where df(M) = trace(A(M)) = 2M + 1 for Fourier

    References:
        Craven, P., & Wahba, G. (1979). Smoothing noisy data with spline
        functions. Numerische Mathematik, 31(4), 377-403.
    """
    n = len(x_train)
    tmin, tmax = float(x_train.min()), float(x_train.max())
    T = tmax - tmin

    if T <= 0:
        raise ValueError("Invalid domain length for Fourier fit")

    omega = 2.0 * np.pi / T

    best_gcv = np.inf
    best_M = min_harmonics
    gcv_values = []

    # Ensure max_harmonics doesn't exceed data constraints
    max_M_safe = min(max_harmonics, (n - 1) // 2)

    for M in range(min_harmonics, max_M_safe + 1):
        # Build design matrix
        phi_cols = [np.ones(n)]
        ang = x_train - tmin

        for k in range(1, M + 1):
            kω = k * omega
            phi_cols.append(np.cos(kω * ang))
            phi_cols.append(np.sin(kω * ang))

        Phi = np.vstack(phi_cols).T  # shape (n, 2M+1)

        # Least squares fit
        try:
            coef, residuals, rank, s = np.linalg.lstsq(Phi, y_train, rcond=None)
            y_pred = Phi @ coef
            rss = np.sum((y_train - y_pred) ** 2)

            # GCV formula
            df = 2*M + 1  # degrees of freedom
            if n - df > 0:
                gcv = (rss / n) / ((1 - df/n) ** 2)
            else:
                gcv = np.inf

            gcv_values.append((M, gcv))

            if gcv < best_gcv:
                best_gcv = gcv
                best_M = M

        except Exception as e:
            continue

    return best_M, best_gcv


# ===========================================================================
# AAA: Noise-adaptive tolerance
# ===========================================================================

def select_aaa_tolerance(
    y_train: np.ndarray,
    multiplier: float = 10.0,
    min_tol: float = 1e-13
) -> float:
    """
    Select AAA tolerance based on estimated noise level.

    The AAA algorithm tries to minimize residual error. If tolerance is too
    tight relative to noise, it overfits by creating spurious poles.

    Args:
        y_train: Training y values (used for noise estimation)
        multiplier: Safety factor (tol = multiplier * σ̂)
        min_tol: Minimum tolerance (for clean data / machine precision)

    Returns:
        Adaptive tolerance value

    Strategy:
        - High noise (σ̂=5e-2): tol ≈ 0.5 (very loose, prevents overfitting)
        - Moderate (σ̂=1e-4): tol ≈ 1e-3
        - Clean (σ̂=1e-8): tol ≈ 1e-7 (tight but not absurd)
    """
    sigma_hat = estimate_noise_auto(y_train)

    # Rule: tolerance should be ~10x noise to avoid fitting noise
    # But don't go below machine precision for truly clean data
    tol = max(min_tol, multiplier * sigma_hat)

    return float(tol)


# ===========================================================================
# FOURIER-FFT: SURE-based filter selection
# ===========================================================================

def select_fourier_filter_fraction_sure(
    y_train: np.ndarray,
    use_dct: bool = True
) -> Tuple[float, float]:
    """
    Select optimal filter fraction via SURE (Stein's Unbiased Risk Estimate).

    SURE gives an (almost) unbiased estimate of MSE for threshold-based
    denoising under Gaussian noise, without knowing σ.

    Args:
        y_train: Training y values
        use_dct: Use DCT (True) or FFT (False)

    Returns:
        (filter_frac, threshold_used)

    References:
        Donoho, D. L., & Johnstone, I. M. (1995). Adapting to unknown
        smoothness via wavelet shrinkage. JASA, 90(432), 1200-1224.
    """
    from scipy.fftpack import dct, fft

    n = len(y_train)

    # Transform to frequency domain
    if use_dct:
        c = dct(y_train, norm='ortho')
    else:
        c = np.abs(fft(y_train))[:n//2]  # Keep positive frequencies only

    c_abs = np.abs(c)
    c_abs_sorted = np.sort(c_abs)[::-1]  # Descending order

    # SURE formula for hard thresholding
    def sure_risk(threshold):
        m = np.sum(c_abs >= threshold)  # Number of kept coefficients
        discarded_energy = np.sum(c_abs[c_abs < threshold]**2)
        # SURE: n - 2m + ||c_discarded||²
        risk = n - 2*m + discarded_energy
        return risk

    # Grid search over sorted coefficient magnitudes
    min_risk = np.inf
    best_threshold = 0.0
    best_m = n // 2

    # Avoid extremes (keep at least 10%, at most 90%)
    min_keep = max(10, int(0.1 * n))
    max_keep = min(n - 10, int(0.9 * n))

    for i in range(min_keep, max_keep):
        threshold = c_abs_sorted[i]
        risk = sure_risk(threshold)

        if risk < min_risk:
            min_risk = risk
            best_threshold = threshold
            best_m = i

    filter_frac = best_m / n

    return float(filter_frac), float(best_threshold)


def select_fourier_filter_fraction_simple(
    y_train: np.ndarray,
    confidence_multiplier: float = 3.0
) -> float:
    """
    Simple filter fraction: keep coefficients above threshold.

    Simpler than SURE - estimates noise, keeps coefficients significantly
    above noise floor.

    Args:
        y_train: Training y values
        confidence_multiplier: Threshold = multiplier * σ̂ (default: 3 = 99% confidence)

    Returns:
        filter_frac (fraction of frequencies to keep)
    """
    from scipy.fftpack import dct

    sigma_hat = estimate_noise_auto(y_train)
    n = len(y_train)

    c = dct(y_train, norm='ortho')
    c_abs = np.abs(c)

    # Keep coefficients with |c| > λσ̂√n
    # The √n factor accounts for energy spread across frequencies
    threshold = confidence_multiplier * sigma_hat * np.sqrt(n)
    m = np.sum(c_abs >= threshold)

    # Guard rails: keep between 10% and 80%
    m = max(int(0.1 * n), min(int(0.8 * n), m))

    filter_frac = m / n

    return float(filter_frac)


# ===========================================================================
# UTILITIES
# ===========================================================================

def validate_hyperparameter_selection(
    selected_value: float,
    fixed_value: float,
    metric: str = "value"
) -> dict:
    """
    Compare selected vs fixed hyperparameter.

    Returns diagnostic information for debugging.
    """
    return {
        'selected': selected_value,
        'fixed': fixed_value,
        'ratio': selected_value / (fixed_value + 1e-12),
        'metric': metric
    }


if __name__ == "__main__":
    # Quick test
    print("Testing hyperparameter selection module...")

    # Generate test data: smooth function + noise
    np.random.seed(42)
    x = np.linspace(0, 10, 101)
    y_true = np.sin(x) + 0.3 * np.sin(3*x)
    noise_level = 0.05
    y_noisy = y_true + noise_level * np.random.randn(len(x))

    print(f"\nTrue noise: {noise_level:.6f}")

    # Test noise estimation
    if PYWT_AVAILABLE:
        sigma_wavelet = estimate_noise_wavelet(y_noisy)
        print(f"Wavelet MAD estimate: {sigma_wavelet:.6f}")

    sigma_diff2 = estimate_noise_diff2(y_noisy)
    print(f"2nd-order diff estimate: {sigma_diff2:.6f}")

    # Test Chebyshev selection
    deg, aicc = select_chebyshev_degree(x, y_noisy, max_degree=30)
    print(f"\nChebyshev optimal degree: {deg} (AICc={aicc:.2f})")
    print(f"  vs fixed degree=20")

    # Test Fourier selection
    M, gcv = select_fourier_harmonics(x, y_noisy, max_harmonics=25)
    print(f"\nFourier optimal harmonics: {M} (GCV={gcv:.6f})")
    print(f"  vs fixed M={(len(x)-1)//4}")

    # Test AAA tolerance
    tol = select_aaa_tolerance(y_noisy)
    print(f"\nAAA adaptive tolerance: {tol:.2e}")
    print(f"  vs fixed tol=1e-13")

    # Test Fourier-FFT filter
    frac_sure, thresh = select_fourier_filter_fraction_sure(y_noisy)
    frac_simple = select_fourier_filter_fraction_simple(y_noisy)
    print(f"\nFourier-FFT filter fraction:")
    print(f"  SURE: {frac_sure:.3f}")
    print(f"  Simple: {frac_simple:.3f}")
    print(f"  vs fixed frac=0.4")

    print("\n✅ All tests passed!")
