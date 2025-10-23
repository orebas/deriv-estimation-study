"""
Spectral Methods for Derivative Estimation

This module implements spectral and Fourier-based derivative estimation methods:
1. Fourier series with fixed harmonics
2. Fourier series with GCV-based adaptive harmonics selection
3. FFT-based spectral differentiation with adaptive noise filtering
4. Fourier continuation (trend removal for non-periodic data)
5. Fourier continuation with adaptive trend degree and harmonics
6. AD-backed trigonometric polynomial using autograd
7. AD-backed trigonometric with GCV harmonics selection
8. FFT-based derivative with taper and regularization
"""

from typing import Dict
import numpy as np
import os

# Import base class from common utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import MethodEvaluator, ADAPTIVE_HYPERPARAMS, AUTOGRAD_AVAILABLE

if ADAPTIVE_HYPERPARAMS:
    from hyperparameters import (
        select_fourier_harmonics,
        select_fourier_filter_fraction_simple,
        select_chebyshev_degree
    )

if AUTOGRAD_AVAILABLE:
    import autograd.numpy as anp
    from autograd import grad as egrad


class SpectralMethods(MethodEvaluator):
    """
    Spectral and Fourier-based methods for derivative estimation.

    Inherits from base MethodEvaluator and implements spectral-specific methods.
    """

    def evaluate_method(self, method_name: str) -> Dict:
        """
        Dispatch to the appropriate spectral method based on method name.

        Args:
            method_name: Name of method to evaluate. Supported:
                - "fourier": Trigonometric polynomial (fixed harmonics)
                - "Fourier-GCV": Fourier with GCV harmonics selection
                - "Fourier-FFT-Adaptive": FFT with adaptive noise filtering
                - "fourier_continuation": Trend-removed Fourier (non-periodic)
                - "Fourier-Continuation-Adaptive": Adaptive trend + harmonics
                - "AD_Trig": AD-backed trigonometric polynomial
                - "AD-Trig-Adaptive": AD-trig with GCV harmonics
                - "SpectralTaperDerivative_Python": FFT with taper/regularization

        Returns:
            Dictionary with predictions, failures, and metadata
        """
        if method_name == "fourier":
            return self._fourier()
        elif method_name == "Fourier-GCV":
            return self._fourier_gcv()
        elif method_name == "Fourier-FFT-Adaptive":
            return self._fourier_fft_adaptive()
        elif method_name == "fourier_continuation":
            return self._fourier_continuation()
        elif method_name == "Fourier-Continuation-Adaptive":
            return self._fourier_continuation_adaptive()
        elif method_name == "AD_Trig":
            return self._ad_trig()
        elif method_name == "AD-Trig-Adaptive":
            return self._ad_trig_adaptive()
        elif method_name == "SpectralTaperDerivative_Python":
            return self._spectral_taper_derivative()
        else:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": f"Unknown spectral method: {method_name}"}
            }

    def _fourier(self) -> Dict:
        """Trigonometric polynomial with analytic n-th derivatives.

        Uses fixed number of harmonics based on training set size.
        For adaptive selection, use Fourier-GCV method instead.

        Returns:
            Dictionary containing:
                predictions: Dict mapping derivative order to list of predictions
                failures: Dict of any errors encountered
                meta: Number of harmonics and selection method
        """
        t = self.x_train
        y = self.y_train
        tmin = float(t.min())
        tmax = float(t.max())
        T = tmax - tmin
        if T <= 0:
            raise ValueError("Invalid domain length for Fourier fit")
        omega = 2.0 * np.pi / T

        n_train = len(t)

        # Use fixed harmonics (original behavior)
        # For adaptive selection, use Fourier-GCV method instead
        M = max(1, min((n_train - 1) // 4, 25))
        gcv = None
        selection_method = "fixed"

        # Build design matrix: [1, cos(k ω (t-tmin)), sin(k ω (t-tmin))] for k=1..M
        phi_cols = [np.ones_like(t)]
        ang = (t - tmin)
        for k in range(1, M + 1):
            kω = k * omega
            phi_cols.append(np.cos(kω * ang))
            phi_cols.append(np.sin(kω * ang))
        Phi = np.vstack(phi_cols).T  # shape (N, 2M+1)

        coef, *_ = np.linalg.lstsq(Phi, y, rcond=None)
        c0 = float(coef[0])
        a = coef[1::2]
        b = coef[2::2]

        predictions = {}
        failures = {}

        ang_eval = (self.x_eval - tmin)
        for order in self.orders:
            try:
                if order == 0:
                    vals = np.full_like(ang_eval, c0, dtype=float)
                    for k in range(1, M + 1):
                        kω = k * omega
                        vals += a[k-1] * np.cos(kω * ang_eval) + b[k-1] * np.sin(kω * ang_eval)
                else:
                    shift = order * (np.pi / 2.0)
                    vals = np.zeros_like(ang_eval, dtype=float)
                    for k in range(1, M + 1):
                        kω = k * omega
                        factor = (kω ** order)
                        θ = kω * ang_eval
                        vals += a[k-1] * factor * np.cos(θ + shift) + b[k-1] * factor * np.sin(θ + shift)
                predictions[order] = [float(v) for v in np.asarray(vals)]
            except Exception as e:
                failures[order] = str(e)
                predictions[order] = [np.nan] * len(self.x_eval)

        meta = {"harmonics": M, "selection": selection_method}
        if gcv is not None:
            meta["gcv"] = float(gcv)

        return {"predictions": predictions, "failures": failures, "meta": meta}

    def _fourier_gcv(self) -> Dict:
        """
        Fourier series with adaptive harmonics selection via GCV.

        Uses Generalized Cross-Validation to automatically select the number
        of harmonics based on bias-variance tradeoff.

        Returns:
            Dictionary containing:
                predictions: Dict mapping derivative order to list of predictions
                failures: Dict of any errors encountered
                meta: Number of harmonics, GCV score, and selection method
        """
        if not ADAPTIVE_HYPERPARAMS:
            return {
                "predictions": {},
                "failures": {0: "hyperparameters module not available"},
                "meta": {"error": "hyperparameters module not found"}
            }

        t = self.x_train
        y = self.y_train
        tmin = float(t.min())
        tmax = float(t.max())
        T = tmax - tmin
        if T <= 0:
            raise ValueError("Invalid domain length for Fourier fit")
        omega = 2.0 * np.pi / T
        n_train = len(t)

        # Always use adaptive harmonics selection via GCV
        M, gcv = select_fourier_harmonics(
            self.x_train, self.y_train,
            max_harmonics=25,
            min_harmonics=1
        )

        # Build design matrix: [1, cos(k ω (t-tmin)), sin(k ω (t-tmin))] for k=1..M
        phi_cols = [np.ones_like(t)]
        ang = (t - tmin)
        for k in range(1, M + 1):
            kω = k * omega
            phi_cols.append(np.cos(kω * ang))
            phi_cols.append(np.sin(kω * ang))
        Phi = np.vstack(phi_cols).T  # shape (N, 2M+1)

        coef, *_ = np.linalg.lstsq(Phi, y, rcond=None)
        c0 = float(coef[0])
        a = coef[1::2]
        b = coef[2::2]

        predictions = {}
        failures = {}

        ang_eval = (self.x_eval - tmin)
        for order in self.orders:
            try:
                if order == 0:
                    vals = np.full_like(ang_eval, c0, dtype=float)
                    for k in range(1, M + 1):
                        kω = k * omega
                        vals += a[k-1] * np.cos(kω * ang_eval) + b[k-1] * np.sin(kω * ang_eval)
                else:
                    shift = order * (np.pi / 2.0)
                    vals = np.zeros_like(ang_eval, dtype=float)
                    for k in range(1, M + 1):
                        kω = k * omega
                        factor = (kω ** order)
                        θ = kω * ang_eval
                        vals += a[k-1] * factor * np.cos(θ + shift) + b[k-1] * factor * np.sin(θ + shift)
                predictions[order] = [float(v) for v in np.asarray(vals)]
            except Exception as e:
                failures[order] = str(e)
                predictions[order] = [np.nan] * len(self.x_eval)

        meta = {"harmonics": M, "selection": "GCV", "gcv": float(gcv)}
        return {"predictions": predictions, "failures": failures, "meta": meta}

    def _fourier_fft_adaptive(self) -> Dict:
        """
        FFT-based spectral differentiation with adaptive noise-based filtering.

        Uses wavelet noise estimation to adaptively select filter fraction,
        preventing noise amplification in high-order derivatives.

        Matches Julia's Fourier-FFT-Adaptive implementation.

        Returns:
            Dictionary containing:
                predictions: Dict mapping derivative order to list of predictions
                failures: Dict of any errors encountered
                meta: Filter fraction, cutoff frequency, and selection method
        """
        if not ADAPTIVE_HYPERPARAMS:
            return {
                "predictions": {},
                "failures": {0: "hyperparameters module not available"},
                "meta": {"error": "hyperparameters module not found"}
            }

        from scipy.fft import fft, ifft, fftfreq
        from scipy.interpolate import interp1d

        x = self.x_train
        y = self.y_train
        n = len(y)

        # Estimate noise and select adaptive filter fraction
        filter_frac = select_fourier_filter_fraction_simple(y, confidence_multiplier=3.0)

        # Compute FFT
        y_fft = fft(y)
        freqs = fftfreq(n, d=float(np.mean(np.diff(x))))
        k = 2.0 * np.pi * freqs  # Angular wavenumbers

        # Determine cutoff frequency based on filter fraction
        k_abs = np.abs(k)
        k_max = np.max(k_abs)
        k_cutoff = filter_frac * k_max

        predictions = {}
        failures = {}

        for order in self.orders:
            try:
                if order == 0:
                    # Order 0: simple interpolation
                    interp = interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')
                    vals = interp(self.x_eval)
                    predictions[order] = [float(v) for v in vals]
                else:
                    # Apply spectral differentiation with filtering
                    deriv_fft = y_fft.copy()

                    # Multiply by (ik)^n and apply low-pass filter
                    for i in range(n):
                        if k_abs[i] <= k_cutoff:
                            deriv_fft[i] *= (1j * k[i]) ** order
                        else:
                            deriv_fft[i] = 0.0  # Zero out high frequencies

                    # Inverse FFT to get derivative
                    deriv = np.real(ifft(deriv_fft))

                    # Interpolate to evaluation points
                    interp = interp1d(x, deriv, kind='linear', bounds_error=False, fill_value='extrapolate')
                    vals = interp(self.x_eval)
                    predictions[order] = [float(v) for v in vals]

            except Exception as e:
                failures[order] = str(e)
                predictions[order] = [np.nan] * len(self.x_eval)

        meta = {
            "filter_frac": float(filter_frac),
            "cutoff_freq": float(k_cutoff / (2.0 * np.pi)),
            "selection": "noise-adaptive"
        }

        return {"predictions": predictions, "failures": failures, "meta": meta}

    def _fourier_continuation(self) -> Dict:
        """Trend-removed trigonometric LS fit with analytic n-th derivatives (non-periodic aid).

        We fit a low-degree polynomial trend and a trig series on residuals; derivatives add.

        Environment variable for tuning:
        - FC_TREND_DEG: Polynomial trend degree (default 3)

        Returns:
            Dictionary containing:
                predictions: Dict mapping derivative order to list of predictions
                failures: Dict of any errors encountered
                meta: Number of harmonics and trend degree
        """
        t = self.x_train
        y = self.y_train
        tmin = float(t.min())
        tmax = float(t.max())
        T = tmax - tmin
        if T <= 0:
            raise ValueError("Invalid domain length for Fourier continuation fit")
        omega = 2.0 * np.pi / T

        # Trend degree from env, default cubic
        deg = int(os.environ.get("FC_TREND_DEG", "3"))
        tt = (t - tmin)
        # Vandermonde [1, tt, tt^2, ...]
        V = np.vander(tt, N=deg + 1, increasing=True)
        coef_trend, *_ = np.linalg.lstsq(V, y, rcond=None)

        # Remove trend
        y_resid = y - V @ coef_trend

        # Trig LS like _fourier
        n_train = len(t)
        M = max(1, min(((n_train - 1) // 4), 25))
        phi_cols = [np.ones_like(t)]
        for k in range(1, M + 1):
            kω = k * omega
            phi_cols.append(np.cos(kω * tt))
            phi_cols.append(np.sin(kω * tt))
        Phi = np.vstack(phi_cols).T
        coef_trig, *_ = np.linalg.lstsq(Phi, y_resid, rcond=None)
        c0 = float(coef_trig[0])
        a = coef_trig[1::2]
        b = coef_trig[2::2]

        predictions = {}
        failures = {}

        tt_eval = (self.x_eval - tmin)
        # Helper: evaluate trend and its derivatives at eval points
        def eval_trend(n_order: int) -> np.ndarray:
            vals = np.zeros_like(tt_eval, dtype=float)
            if n_order > deg:
                return vals
            # Derivative coefficients
            coeffs = coef_trend.copy()
            for _ in range(n_order):
                coeffs = np.array([i * coeffs[i] for i in range(1, len(coeffs))], dtype=float)
            # Evaluate
            for i, c in enumerate(coeffs):
                vals += c * (tt_eval ** i)
            return vals

        for order in self.orders:
            try:
                # Trend part
                vals = eval_trend(order)
                # Trig part
                if order == 0:
                    vals += c0
                    for k in range(1, M + 1):
                        kω = k * omega
                        vals += a[k-1] * np.cos(kω * tt_eval) + b[k-1] * np.sin(kω * tt_eval)
                else:
                    shift = order * (np.pi / 2.0)
                    for k in range(1, M + 1):
                        kω = k * omega
                        factor = (kω ** order)
                        θ = kω * tt_eval
                        vals += a[k-1] * factor * np.cos(θ + shift) + b[k-1] * factor * np.sin(θ + shift)
                predictions[order] = [float(v) for v in np.asarray(vals)]
            except Exception as e:
                failures[order] = str(e)
                predictions[order] = [np.nan] * len(self.x_eval)

        return {"predictions": predictions, "failures": failures, "meta": {"harmonics": M, "trend_deg": deg}}

    def _fourier_continuation_adaptive(self) -> Dict:
        """
        Fourier continuation with adaptive trend degree (AICc) and harmonics (GCV).

        Uses Chebyshev AICc for trend removal and Fourier GCV for residuals,
        providing automatic hyperparameter selection for non-periodic data.

        Returns:
            Dictionary containing:
                predictions: Dict mapping derivative order to list of predictions
                failures: Dict of any errors encountered
                meta: Harmonics, trend degree, selection methods, and scores
        """
        if not ADAPTIVE_HYPERPARAMS:
            return {
                "predictions": {},
                "failures": {0: "hyperparameters module not available"},
                "meta": {"error": "hyperparameters module not found"}
            }

        t = self.x_train
        y = self.y_train
        tmin = float(t.min())
        tmax = float(t.max())
        T = tmax - tmin
        if T <= 0:
            raise ValueError("Invalid domain length for Fourier continuation fit")
        omega = 2.0 * np.pi / T

        # 1. Adaptive trend degree selection via AICc
        deg, aicc = select_chebyshev_degree(t, y, max_degree=5, min_degree=1)

        tt = (t - tmin)
        # Vandermonde [1, tt, tt^2, ...]
        V = np.vander(tt, N=deg + 1, increasing=True)
        coef_trend, *_ = np.linalg.lstsq(V, y, rcond=None)

        # Remove trend
        y_resid = y - V @ coef_trend

        # 2. Adaptive harmonics selection via GCV on residuals
        M, gcv = select_fourier_harmonics(t, y_resid, max_harmonics=25, min_harmonics=1)

        # Fit Fourier to residuals
        phi_cols = [np.ones_like(t)]
        for k in range(1, M + 1):
            kω = k * omega
            phi_cols.append(np.cos(kω * tt))
            phi_cols.append(np.sin(kω * tt))
        Phi = np.vstack(phi_cols).T
        coef_trig, *_ = np.linalg.lstsq(Phi, y_resid, rcond=None)
        c0 = float(coef_trig[0])
        a = coef_trig[1::2]
        b = coef_trig[2::2]

        predictions = {}
        failures = {}

        tt_eval = (self.x_eval - tmin)

        # Helper: evaluate trend and its derivatives at eval points
        def eval_trend(n_order: int) -> np.ndarray:
            vals = np.zeros_like(tt_eval, dtype=float)
            if n_order > deg:
                return vals
            # Derivative coefficients
            coeffs = coef_trend.copy()
            for _ in range(n_order):
                coeffs = np.array([i * coeffs[i] for i in range(1, len(coeffs))], dtype=float)
            # Evaluate
            for i, c in enumerate(coeffs):
                vals += c * (tt_eval ** i)
            return vals

        for order in self.orders:
            try:
                # Trend part
                vals = eval_trend(order)
                # Trig part
                if order == 0:
                    vals += c0
                    for k in range(1, M + 1):
                        kω = k * omega
                        vals += a[k-1] * np.cos(kω * tt_eval) + b[k-1] * np.sin(kω * tt_eval)
                else:
                    shift = order * (np.pi / 2.0)
                    for k in range(1, M + 1):
                        kω = k * omega
                        factor = (kω ** order)
                        θ = kω * tt_eval
                        vals += a[k-1] * factor * np.cos(θ + shift) + b[k-1] * factor * np.sin(θ + shift)
                predictions[order] = [float(v) for v in np.asarray(vals)]
            except Exception as e:
                failures[order] = str(e)
                predictions[order] = [np.nan] * len(self.x_eval)

        meta = {
            "harmonics": M,
            "trend_deg": deg,
            "selection_trend": "AICc",
            "selection_harmonics": "GCV",
            "aicc": float(aicc),
            "gcv": float(gcv)
        }

        return {"predictions": predictions, "failures": failures, "meta": meta}

    def _ad_trig(self) -> Dict:
        """AD-backed trigonometric polynomial using autograd for derivatives.

        Uses automatic differentiation to compute derivatives of all orders,
        with harmonics selected based on training set size.

        Returns:
            Dictionary containing:
                predictions: Dict mapping derivative order to list of predictions
                failures: Dict of any errors encountered
                meta: Number of harmonics used
        """
        if not AUTOGRAD_AVAILABLE:
            raise RuntimeError("autograd is not available")

        t = self.x_train
        y = self.y_train
        tmin = float(t.min())
        tmax = float(t.max())
        T = tmax - tmin
        if T <= 0:
            raise ValueError("Invalid domain length for AD-trig fit")
        omega = 2.0 * np.pi / T

        n_train = len(t)
        M = max(1, min( (n_train - 1) // 4, 20 ))

        # LS fit coefficients (NumPy OK; coefficients are constants for AD eval)
        phi_cols = [np.ones_like(t)]
        ang = (t - tmin)
        for k in range(1, M + 1):
            kω = k * omega
            phi_cols.append(np.cos(kω * ang))
            phi_cols.append(np.sin(kω * ang))
        Phi = np.vstack(phi_cols).T
        coef, *_ = np.linalg.lstsq(Phi, y, rcond=None)
        c0 = float(coef[0])
        a = coef[1::2]
        b = coef[2::2]

        # Define autograd-evaluable model
        a_ag = anp.array(a)
        b_ag = anp.array(b)
        def f_scalar(tt):
            θ = anp.arange(1, M + 1) * omega * (tt - tmin)
            return c0 + anp.sum(a_ag * anp.cos(θ) + b_ag * anp.sin(θ))

        predictions = {}
        failures = {}

        for order in self.orders:
            try:
                if order == 0:
                    vals = np.array([float(f_scalar(tt)) for tt in self.x_eval])
                else:
                    g = f_scalar
                    for _ in range(order):
                        g = egrad(g)
                    vals = np.array([float(g(tt)) for tt in self.x_eval])
                predictions[order] = [float(v) for v in np.asarray(vals)]
            except Exception as e:
                failures[order] = str(e)
                predictions[order] = [np.nan] * len(self.x_eval)

        return {"predictions": predictions, "failures": failures, "meta": {"harmonics": M}}

    def _ad_trig_adaptive(self) -> Dict:
        """
        AD-backed trigonometric polynomial with GCV-selected harmonics.

        Uses Generalized Cross-Validation to adaptively select the number
        of harmonics, then uses autograd for automatic differentiation.

        Returns:
            Dictionary containing:
                predictions: Dict mapping derivative order to list of predictions
                failures: Dict of any errors encountered
                meta: Number of harmonics, GCV score, and selection method
        """
        if not AUTOGRAD_AVAILABLE:
            raise RuntimeError("autograd is not available")

        if not ADAPTIVE_HYPERPARAMS:
            return {
                "predictions": {},
                "failures": {0: "hyperparameters module not available"},
                "meta": {"error": "hyperparameters module not found"}
            }

        t = self.x_train
        y = self.y_train
        tmin = float(t.min())
        tmax = float(t.max())
        T = tmax - tmin
        if T <= 0:
            raise ValueError("Invalid domain length for AD-trig fit")
        omega = 2.0 * np.pi / T

        # Adaptive harmonics selection via GCV
        M, gcv = select_fourier_harmonics(
            self.x_train, self.y_train,
            max_harmonics=25,
            min_harmonics=1
        )

        # LS fit coefficients (NumPy OK; coefficients are constants for AD eval)
        phi_cols = [np.ones_like(t)]
        ang = (t - tmin)
        for k in range(1, M + 1):
            kω = k * omega
            phi_cols.append(np.cos(kω * ang))
            phi_cols.append(np.sin(kω * ang))
        Phi = np.vstack(phi_cols).T
        coef, *_ = np.linalg.lstsq(Phi, y, rcond=None)
        c0 = float(coef[0])
        a = coef[1::2]
        b = coef[2::2]

        # Define autograd-evaluable model
        a_ag = anp.array(a)
        b_ag = anp.array(b)
        def f_scalar(tt):
            θ = anp.arange(1, M + 1) * omega * (tt - tmin)
            return c0 + anp.sum(a_ag * anp.cos(θ) + b_ag * anp.sin(θ))

        predictions = {}
        failures = {}

        for order in self.orders:
            try:
                if order == 0:
                    vals = np.array([float(f_scalar(tt)) for tt in self.x_eval])
                else:
                    g = f_scalar
                    for _ in range(order):
                        g = egrad(g)
                    vals = np.array([float(g(tt)) for tt in self.x_eval])
                predictions[order] = [float(v) for v in np.asarray(vals)]
            except Exception as e:
                failures[order] = str(e)
                predictions[order] = [np.nan] * len(self.x_eval)

        return {"predictions": predictions, "failures": failures, "meta": {"harmonics": M, "selection": "GCV", "gcv": float(gcv)}}

    def _spectral_taper_derivative(self) -> Dict:
        """FFT-based derivative with taper and low-pass regularization.

        Steps: detrend (optional), apply taper (Tukey), FFT, multiply by (i*k)^n,
        apply spectral shrinkage, inverse FFT, and sample at original grid.

        Environment variables for tuning:
        - SPEC_TAPER_ALPHA: Tukey window alpha parameter (default 0.25)
        - SPEC_CUTOFF: Fraction of Nyquist frequency for cutoff (default 0.5)
        - SPEC_SHRINK: Spectral shrinkage factor 0..1 (default 0.0)

        Returns:
            Dictionary containing:
                predictions: Dict mapping derivative order to list of predictions
                failures: Dict of any errors encountered
                meta: Taper alpha, cutoff, and shrinkage parameters
        """
        t = self.x_train
        y = self.y_train
        n = len(t)
        if n < 8:
            raise ValueError("Not enough points for spectral derivative")
        dt = float(np.mean(np.diff(t)))

        # Taper (Tukey)
        alpha = float(os.environ.get("SPEC_TAPER_ALPHA", "0.25"))
        m = np.arange(n)
        # Tukey window
        def tukey(M, alpha):
            if alpha <= 0:
                return np.ones(M)
            if alpha >= 1:
                return np.hanning(M)
            w = np.ones(M)
            p = int(np.floor(alpha * (M - 1) / 2.0))
            w[:p] = 0.5 * (1 + np.cos(np.pi * (2 * m[:p] / (alpha * (M - 1)) - 1)))
            w[-p:] = 0.5 * (1 + np.cos(np.pi * (2 * m[:p][::-1] / (alpha * (M - 1)) - 1)))
            return w
        w = tukey(n, alpha)
        yw = y * w

        # FFT frequencies
        freqs = np.fft.fftfreq(n, d=dt)
        k = 2.0 * np.pi * freqs
        Y = np.fft.fft(yw)

        cutoff = float(os.environ.get("SPEC_CUTOFF", "0.5"))  # fraction of Nyquist
        shrink = float(os.environ.get("SPEC_SHRINK", "0.0"))  # 0..1
        k_nyq = np.pi / dt
        mask = np.abs(k) <= cutoff * k_nyq

        predictions = {}
        failures = {}
        for order in self.orders:
            try:
                if order == 0:
                    vals = y
                else:
                    # spectral differentiation: (i k)^order
                    factor = (1j * k) ** order
                    Z = Y * factor
                    # low-pass taper
                    Z = np.where(mask, Z * (1.0 - shrink), 0.0)
                    z_time = np.fft.ifft(Z)
                    vals = np.real(z_time) / np.maximum(w, 1e-6)  # de-taper
                predictions[order] = [float(v) for v in np.asarray(vals)]
            except Exception as e:
                failures[order] = str(e)
                predictions[order] = [np.nan] * len(self.x_eval)
        return {"predictions": predictions, "failures": failures, "meta": {"alpha": alpha, "cutoff": cutoff, "shrink": shrink}}
