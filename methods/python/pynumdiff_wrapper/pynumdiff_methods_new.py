"""
PyNumDiff-Based Methods for Derivative Estimation

This module implements derivative estimation methods using the PyNumDiff package.
PyNumDiff provides numerical differentiation of noisy time-series data with automatic
parameter selection.

Methods organized by derivative order support:
1. Full orders 0-7 support (native higher-order derivatives):
   - Savitzky-Golay filter (uses scipy.signal.savgol_filter with deriv parameter)
   - Spectral method (FFT domain multiplication by (iω)^n)

2. Orders 0-1 only (PyNumDiff returns only first derivative):
   - Butterworth filtering
   - Spline smoothing
   - Gaussian kernel smoothing
   - Friedrichs mollification
   - Kalman RTS smoother
   - Total Variation regularization

IMPORTANT: We do NOT fit splines to x_smooth for higher derivatives, as this
introduces unnecessary interpolation error. Methods either support higher orders
natively or report only orders 0-1.

Reference:
- PyNumDiff documentation: https://github.com/florisvb/PyNumDiff
- JOSS paper: Djerroud et al. (2022)
"""

from typing import Dict, Optional
import numpy as np
import warnings
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import UnivariateSpline

# Import PyNumDiff methods
try:
    import pynumdiff
    from pynumdiff import smooth_finite_difference as sfd
    from pynumdiff import total_variation_regularization as tvr
    from pynumdiff import kalman_smooth
    from pynumdiff.polynomial_fit import savgoldiff as pnd_savgoldiff
    from pynumdiff.basis_fit import spectraldiff as pnd_spectraldiff
    from pynumdiff.optimize import optimize as pnd_optimize
    PYNUMDIFF_AVAILABLE = True
except ImportError:
    PYNUMDIFF_AVAILABLE = False
    warnings.warn("PyNumDiff not available - install via: pip install pynumdiff")

# Import base class from common utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import MethodEvaluator


class PyNumDiffMethods(MethodEvaluator):
    """
    PyNumDiff-based derivative estimation methods.

    Inherits from base MethodEvaluator and implements PyNumDiff-specific methods.
    """

    def evaluate_method(self, method_name: str) -> Dict:
        """
        Dispatch to the appropriate PyNumDiff method based on method name.

        Args:
            method_name: Name of method to evaluate. Supported:
                Full orders 0-7 support:
                - "PyNumDiff-SavGol-Auto": Savitzky-Golay with automatic parameters
                - "PyNumDiff-SavGol-Tuned": Savitzky-Golay with tuned parameters
                - "PyNumDiff-Spectral-Auto": Spectral method with automatic parameters
                - "PyNumDiff-Spectral-Tuned": Spectral method with tuned parameters

                Orders 0-1 only:
                - "PyNumDiff-Butter-Auto": Butterworth with automatic parameters
                - "PyNumDiff-Butter-Tuned": Butterworth with tuned parameters
                - "PyNumDiff-Gaussian-Auto": Gaussian kernel with automatic parameters
                - "PyNumDiff-Gaussian-Tuned": Gaussian kernel with tuned parameters
                - "PyNumDiff-Friedrichs-Auto": Friedrichs with automatic parameters
                - "PyNumDiff-Friedrichs-Tuned": Friedrichs with tuned parameters
                - "PyNumDiff-Kalman-Auto": Kalman RTS with automatic parameters
                - "PyNumDiff-Kalman-Tuned": Kalman RTS with tuned parameters
                - "PyNumDiff-TV-Velocity": TV for 1st derivative only
                - "PyNumDiff-TV-Acceleration": TV for 2nd derivative only (reports 0-1)
                - "PyNumDiff-TV-Jerk": TV for 3rd derivative only (reports 0-1)

        Returns:
            Dictionary with predictions, failures, and metadata
        """
        if not PYNUMDIFF_AVAILABLE:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": "PyNumDiff package not available"}
            }

        # Full orders 0-7 support
        if method_name == "PyNumDiff-SavGol-Auto":
            return self._savgol(regime="auto")
        elif method_name == "PyNumDiff-SavGol-Tuned":
            return self._savgol(regime="tuned")
        elif method_name == "PyNumDiff-Spectral-Auto":
            return self._spectral(regime="auto")
        elif method_name == "PyNumDiff-Spectral-Tuned":
            return self._spectral(regime="tuned")

        # Orders 0-1 only methods
        elif method_name == "PyNumDiff-Butter-Auto":
            return self._butterdiff(regime="auto")
        elif method_name == "PyNumDiff-Butter-Tuned":
            return self._butterdiff(regime="tuned")
        elif method_name == "PyNumDiff-Gaussian-Auto":
            return self._gaussiandiff(regime="auto")
        elif method_name == "PyNumDiff-Gaussian-Tuned":
            return self._gaussiandiff(regime="tuned")
        elif method_name == "PyNumDiff-Friedrichs-Auto":
            return self._friedrichsdiff(regime="auto")
        elif method_name == "PyNumDiff-Friedrichs-Tuned":
            return self._friedrichsdiff(regime="tuned")
        elif method_name == "PyNumDiff-Kalman-Auto":
            return self._kalman_smooth(regime="auto")
        elif method_name == "PyNumDiff-Kalman-Tuned":
            return self._kalman_smooth(regime="tuned")
        elif method_name == "PyNumDiff-TV-Velocity":
            return self._tv_velocity()
        elif method_name == "PyNumDiff-TV-Acceleration":
            return self._tv_acceleration()
        elif method_name == "PyNumDiff-TV-Jerk":
            return self._tv_jerk()
        else:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": f"Unknown PyNumDiff method: {method_name}"}
            }

    # =========================================================================
    # FULL ORDERS 0-7 SUPPORT
    # =========================================================================

    def _savgol(self, regime: str = "auto") -> Dict:
        """
        Savitzky-Golay filter with native support for orders 0-7.

        Uses scipy.signal.savgol_filter with deriv=0,1,2,...,7 parameter.
        No spline fitting - uses native polynomial differentiation.

        Args:
            regime: "auto" for automatic parameters or "tuned" for conservative values

        Returns:
            Dictionary with predictions, failures, and metadata
        """
        y = self.y_train
        n = len(y)
        dt = float(np.mean(np.diff(self.x_train))) if n > 1 else 1.0

        # Determine parameters
        if regime == "auto":
            # Use PyNumDiff's optimize for window and degree
            try:
                from pynumdiff.polynomial_fit import savgoldiff
                opt = self._optimize_params(savgoldiff, y, dt)
                if opt is None:
                    return {"predictions": {}, "failures": {"error": "savgol_optimize_failed"}}

                # Extract window_size and degree from optimized params
                window_size = opt["params"].get("window_size", 15)
                degree = opt["params"].get("degree", 7)
            except Exception as e:
                return {"predictions": {}, "failures": {"error": f"savgol_auto_failed: {e}"}}
        else:  # tuned
            # Conservative parameters
            window_size = min(15, n if n % 2 == 1 else n - 1)
            degree = 7  # Need degree >= max_order

        # Ensure window_size is odd and reasonable
        if window_size % 2 == 0:
            window_size += 1
        window_size = min(window_size, n)
        window_size = max(window_size, degree + 2)  # Minimum window for given degree

        predictions = {}
        failures = {}

        # Compute each derivative order using native deriv parameter
        for order in self.orders:
            try:
                if order > degree:
                    # Can't compute derivatives higher than polynomial degree
                    failures[order] = f"order {order} exceeds degree {degree}"
                    predictions[order] = [np.nan] * len(self.x_eval)
                    continue

                # Use scipy's savgol_filter with native deriv parameter
                y_deriv_train = savgol_filter(y, window_size, degree, deriv=order, delta=dt)

                # Interpolate to eval points
                spline = UnivariateSpline(self.x_train, y_deriv_train, k=3, s=0)
                predictions[order] = [float(v) for v in spline(self.x_eval)]

            except Exception as e:
                failures[order] = str(e)
                predictions[order] = [np.nan] * len(self.x_eval)

        meta = {
            "method": "savgol",
            "regime": regime,
            "window_size": window_size,
            "degree": degree,
            "dt": dt
        }

        return {
            "predictions": predictions,
            "failures": failures,
            "meta": meta
        }

    def _spectral(self, regime: str = "auto") -> Dict:
        """
        Spectral differentiation using FFT with native support for orders 0-7.

        Computes nth derivative by multiplying by (iω)^n in frequency domain.
        No interpolation - exact in Fourier domain (up to FFT accuracy).

        Args:
            regime: "auto" for automatic parameters or "tuned" for conservative values

        Returns:
            Dictionary with predictions, failures, and metadata
        """
        y = self.y_train
        t = self.x_train
        n = len(y)
        dt = float(np.mean(np.diff(t))) if n > 1 else 1.0

        # Determine high frequency cutoff
        if regime == "auto":
            try:
                opt = self._optimize_params(pnd_spectraldiff, y, dt)
                if opt is None:
                    return {"predictions": {}, "failures": {"error": "spectral_optimize_failed"}}
                high_freq_cutoff = opt["params"].get("high_freq_cutoff", 0.1)
                even_extension = opt["params"].get("even_extension", True)
                pad_to_zero_dxdt = opt["params"].get("pad_to_zero_dxdt", True)
            except Exception as e:
                return {"predictions": {}, "failures": {"error": f"spectral_auto_failed: {e}"}}
        else:  # tuned
            high_freq_cutoff = 0.1
            even_extension = True
            pad_to_zero_dxdt = True

        predictions = {}
        failures = {}

        # Compute each derivative order
        for order in self.orders:
            try:
                y_deriv_train = self._spectraldiff_order_n(
                    y, dt, order,
                    high_freq_cutoff=high_freq_cutoff,
                    even_extension=even_extension,
                    pad_to_zero_dxdt=pad_to_zero_dxdt
                )

                # Interpolate to eval points
                spline = UnivariateSpline(t, y_deriv_train, k=3, s=0)
                predictions[order] = [float(v) for v in spline(self.x_eval)]

            except Exception as e:
                failures[order] = str(e)
                predictions[order] = [np.nan] * len(self.x_eval)

        meta = {
            "method": "spectral",
            "regime": regime,
            "high_freq_cutoff": high_freq_cutoff,
            "even_extension": even_extension,
            "pad_to_zero_dxdt": pad_to_zero_dxdt,
            "dt": dt
        }

        return {
            "predictions": predictions,
            "failures": failures,
            "meta": meta
        }

    def _spectraldiff_order_n(self, x, dt, order, high_freq_cutoff=0.1,
                              even_extension=True, pad_to_zero_dxdt=True):
        """
        Compute nth derivative using FFT (modified from PyNumDiff source).

        For nth derivative, multiply by (iω)^n in frequency domain.
        """
        L = len(x)

        # Padding (from PyNumDiff)
        if pad_to_zero_dxdt:
            padding = 100
            pre = x[0] * np.ones(padding)
            post = x[-1] * np.ones(padding)
            x = np.hstack((pre, x, post))
            # Smooth edges (simplified)
            from scipy.ndimage import uniform_filter1d
            x_padded = x.copy()
            x_padded[:padding] = uniform_filter1d(x[:padding*2], size=padding//2)[-padding:]
            x_padded[-padding:] = uniform_filter1d(x[-padding*2:], size=padding//2)[:padding]
            x_padded[padding:-padding] = x[padding:-padding]
            x = x_padded
        else:
            padding = 0

        # Even extension (from PyNumDiff)
        if even_extension:
            x = np.hstack((x, x[::-1]))

        N = len(x)

        # Frequency domain (from PyNumDiff)
        k = np.concatenate((np.arange(N//2 + 1), np.arange(-N//2 + 1, 0)))
        if N % 2 == 0:
            k[N//2] = 0
        omega = k * 2 * np.pi / (dt * N)

        # High-frequency cutoff (from PyNumDiff)
        discrete_cutoff = int(high_freq_cutoff * N / 2)
        omega[discrete_cutoff:N-discrete_cutoff] = 0

        # Nth derivative = multiply by (iω)^n
        # (iω)^n = i^n * ω^n
        # i^0=1, i^1=i, i^2=-1, i^3=-i, i^4=1, ...
        i_power = (1j)**order
        fft_x = np.fft.fft(x)
        fft_deriv = (i_power * omega**order) * fft_x
        deriv = np.real(np.fft.ifft(fft_deriv))

        # Extract original region
        deriv = deriv[padding:L+padding]

        return deriv

    # =========================================================================
    # ORDERS 0-1 ONLY (PyNumDiff returns only first derivative)
    # =========================================================================

    def _orders_0_1_only(self, x_smooth: np.ndarray, dx_smooth: np.ndarray) -> Dict:
        """
        Helper to return only orders 0 and 1, with NaN for higher orders.

        Args:
            x_smooth: Smoothed signal (order 0)
            dx_smooth: First derivative (order 1)

        Returns:
            Dictionary with predictions for orders 0-1, NaN for 2-7
        """
        t = self.x_train
        predictions = {}
        failures = {}

        for order in self.orders:
            try:
                if order == 0:
                    # Order 0: smoothed signal
                    spline = UnivariateSpline(t, x_smooth, k=3, s=0)
                    predictions[order] = [float(v) for v in spline(self.x_eval)]
                elif order == 1:
                    # Order 1: first derivative
                    spline = UnivariateSpline(t, dx_smooth, k=3, s=0)
                    predictions[order] = [float(v) for v in spline(self.x_eval)]
                else:
                    # Orders 2+: not supported by PyNumDiff
                    failures[order] = "PyNumDiff only returns first derivative"
                    predictions[order] = [np.nan] * len(self.x_eval)
            except Exception as e:
                failures[order] = str(e)
                predictions[order] = [np.nan] * len(self.x_eval)

        return {"predictions": predictions, "failures": failures}

    def _butterdiff(self, regime: str = "auto") -> Dict:
        """Butterworth filtering (orders 0-1 only)."""
        t = self.x_train
        y = self.y_train
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 1.0

        try:
            if regime == "auto":
                opt = self._optimize_params(sfd.butterdiff, y, dt)
                if opt is None:
                    return {"predictions": {}, "failures": {"error": "butterdiff_optimize_failed"}}
                x_smooth, dx_dt = sfd.butterdiff(y, dt, **opt["params"])
                meta = {"method": "butterdiff", "regime": regime, "dt": dt,
                        "opt_params": opt["params"], "opt_value": opt["value"], "tvgamma": opt["tvgamma"]}
            else:  # tuned
                cutoff_norm = 0.2
                x_smooth, dx_dt = sfd.butterdiff(y, dt, filter_order=2, cutoff_freq=cutoff_norm)
                meta = {"method": "butterdiff", "regime": regime, "dt": dt, "cutoff_freq": cutoff_norm}

            result = self._orders_0_1_only(x_smooth, dx_dt)
            return {"predictions": result["predictions"], "failures": result["failures"], "meta": meta}

        except Exception as e:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": str(e)}
            }

    def _gaussiandiff(self, regime: str = "auto") -> Dict:
        """Gaussian kernel smoothing (orders 0-1 only)."""
        t = self.x_train
        y = self.y_train
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 1.0

        try:
            if regime == "auto":
                opt = self._optimize_params(sfd.gaussiandiff, y, dt)
                if opt is None:
                    return {"predictions": {}, "failures": {"error": "gaussiandiff_optimize_failed"}}
                x_smooth, dx_dt = sfd.gaussiandiff(y, dt, **opt["params"])
                meta = {"method": "gaussiandiff", "regime": regime, "dt": dt,
                        "opt_params": opt["params"], "opt_value": opt["value"], "tvgamma": opt["tvgamma"]}
            else:  # tuned
                window = min(11, n if n % 2 == 1 else n - 1)
                x_smooth, dx_dt = sfd.gaussiandiff(y, dt, window_size=window)
                meta = {"method": "gaussiandiff", "regime": regime, "dt": dt, "window_size": window}

            result = self._orders_0_1_only(x_smooth, dx_dt)
            return {"predictions": result["predictions"], "failures": result["failures"], "meta": meta}

        except Exception as e:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": str(e)}
            }

    def _friedrichsdiff(self, regime: str = "auto") -> Dict:
        """Friedrichs mollification (orders 0-1 only)."""
        t = self.x_train
        y = self.y_train
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 1.0

        try:
            if regime == "auto":
                opt = self._optimize_params(sfd.friedrichsdiff, y, dt)
                if opt is None:
                    return {"predictions": {}, "failures": {"error": "friedrichsdiff_optimize_failed"}}
                x_smooth, dx_dt = sfd.friedrichsdiff(y, dt, **opt["params"])
                meta = {"method": "friedrichsdiff", "regime": regime, "dt": dt,
                        "opt_params": opt["params"], "opt_value": opt["value"], "tvgamma": opt["tvgamma"]}
            else:  # tuned
                window = min(11, n if n % 2 == 1 else n - 1)
                x_smooth, dx_dt = sfd.friedrichsdiff(y, dt, window_size=window)
                meta = {"method": "friedrichsdiff", "regime": regime, "dt": dt, "window_size": window}

            result = self._orders_0_1_only(x_smooth, dx_dt)
            return {"predictions": result["predictions"], "failures": result["failures"], "meta": meta}

        except Exception as e:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": str(e)}
            }

    def _kalman_smooth(self, regime: str = "auto") -> Dict:
        """Kalman RTS smoother (orders 0-1 only)."""
        t = self.x_train
        y = self.y_train
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 1.0

        try:
            if regime == "auto":
                opt = self._optimize_params(kalman_smooth.rtsdiff, y, dt, maxiter=30)
                if opt is None:
                    return {"predictions": {}, "failures": {"error": "kalman_rtsdiff_optimize_failed"}}
                x_smooth, dx_smooth = kalman_smooth.rtsdiff(y, dt, **opt["params"])
                meta = {"method": "kalman_smooth/rtsdiff", "regime": regime, "dt": dt,
                        "opt_params": opt["params"], "opt_value": opt["value"], "tvgamma": opt["tvgamma"]}
            else:  # tuned
                x_smooth, dx_smooth = kalman_smooth.constant_acceleration(y, dt, r=1e-8, q=1e-6)
                meta = {"method": "kalman_smooth", "regime": regime, "dt": dt, "r": 1e-8, "q": 1e-6}

            result = self._orders_0_1_only(x_smooth, dx_smooth)
            return {"predictions": result["predictions"], "failures": result["failures"], "meta": meta}

        except Exception as e:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": str(e)}
            }

    def _tv_velocity(self) -> Dict:
        """Total Variation regularization for velocity (orders 0-1 only)."""
        t = self.x_train
        y = self.y_train
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 1.0

        try:
            opt = self._optimize_params(tvr.velocity, y, dt)
            if opt is None:
                return {"predictions": {}, "failures": {"error": "tv_velocity_optimize_failed"}}
            x_smooth, dx_dt = tvr.velocity(y, dt, **opt["params"])

            result = self._orders_0_1_only(x_smooth, dx_dt)
            meta = {"method": "tv_velocity", "dt": dt, "opt_params": opt["params"],
                    "opt_value": opt["value"], "tvgamma": opt["tvgamma"]}
            return {"predictions": result["predictions"], "failures": result["failures"], "meta": meta}

        except Exception as e:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": str(e)}
            }

    def _tv_acceleration(self) -> Dict:
        """Total Variation regularization for acceleration (orders 0-1 only)."""
        t = self.x_train
        y = self.y_train
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 1.0

        try:
            opt = self._optimize_params(tvr.acceleration, y, dt)
            if opt is None:
                return {"predictions": {}, "failures": {"error": "tv_acceleration_optimize_failed"}}
            x_smooth, ddx_dt = tvr.acceleration(y, dt, **opt["params"])

            result = self._orders_0_1_only(x_smooth, ddx_dt)
            meta = {"method": "tv_acceleration", "dt": dt, "opt_params": opt["params"],
                    "opt_value": opt["value"], "tvgamma": opt["tvgamma"]}
            return {"predictions": result["predictions"], "failures": result["failures"], "meta": meta}

        except Exception as e:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": str(e)}
            }

    def _tv_jerk(self) -> Dict:
        """Total Variation regularization for jerk (orders 0-1 only)."""
        t = self.x_train
        y = self.y_train
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 1.0

        try:
            opt = self._optimize_params(tvr.jerk, y, dt)
            if opt is None:
                return {"predictions": {}, "failures": {"error": "tv_jerk_optimize_failed"}}
            x_smooth, dddx_dt = tvr.jerk(y, dt, **opt["params"])

            result = self._orders_0_1_only(x_smooth, dddx_dt)
            meta = {"method": "tv_jerk", "dt": dt, "opt_params": opt.get("params", {}),
                    "opt_value": opt.get("value", None), "tvgamma": opt.get("tvgamma", None)}
            return {"predictions": result["predictions"], "failures": result["failures"], "meta": meta}

        except Exception as e:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": str(e)}
            }

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _estimate_cutoff_frequency(self, y: np.ndarray, dt: float) -> float:
        """Estimate cutoff frequency for automatic parameter selection."""
        try:
            t_total = float(self.x_train[-1] - self.x_train[0]) if len(self.x_train) > 1 else (len(y) - 1) * dt
            if t_total > 0:
                scale = np.std(y) if np.std(y) > 0 else (np.max(y) - np.min(y) + 1e-9)
                peaks, _ = find_peaks(y, prominence=max(1e-3 * scale, 1e-9))
                peak_freq = float(len(peaks)) / t_total if t_total > 0 else 0.0
                if peak_freq > 0:
                    f_max = 0.5 / dt
                    peak_freq = max(0.01 * f_max, min(peak_freq, 0.3 * f_max))
                    return peak_freq
        except Exception:
            pass

        # Fallback: use FFT
        n = len(y)
        if n < 10:
            return 0.1

        yf = np.fft.fft(y - np.mean(y))
        freqs = np.fft.fftfreq(n, dt)
        power = np.abs(yf)**2

        sorted_indices = np.argsort(power)[::-1]
        cumsum = np.cumsum(power[sorted_indices])
        idx_90 = np.searchsorted(cumsum, 0.9 * cumsum[-1])
        f_cutoff = np.abs(freqs[sorted_indices[idx_90]])

        f_max = 0.5 / dt
        f_cutoff = max(0.01 * f_max, min(f_cutoff, 0.3 * f_max))

        return f_cutoff

    def _compute_tvgamma_from_cutoff(self, cutoff_frequency: float, dt: float) -> float:
        """Compute tvgamma using PyNumDiff's documented heuristic."""
        try:
            log_gamma = -1.6 * np.log(cutoff_frequency) - 0.71 * np.log(dt) - 5.1
            tvgamma = float(np.exp(log_gamma))
            return max(1e-8, min(tvgamma, 1e6))
        except Exception:
            return 1e-2

    def _tvgamma_docs_heuristic(self, y: np.ndarray, dt: float) -> float:
        """Estimate tvgamma via docs heuristic by first estimating cutoff frequency."""
        f_cutoff = self._estimate_cutoff_frequency(y, dt)
        return self._compute_tvgamma_from_cutoff(f_cutoff, dt)

    def _optimize_params(self,
                         method_func,
                         y: np.ndarray,
                         dt: float,
                         search_space_updates: Optional[Dict] = None,
                         metric: str = "rmse",
                         maxiter: int = 20) -> Optional[Dict]:
        """
        Call PyNumDiff's optimize() with tvgamma heuristic to get best params.
        Returns dict with keys: params, value. Returns None on failure.
        """
        try:
            tvgamma = self._tvgamma_docs_heuristic(y, dt)
            ssu = search_space_updates or {}
            params, val = pnd_optimize(method_func, y, dt,
                                       tvgamma=tvgamma,
                                       search_space_updates=ssu,
                                       metric=metric,
                                       padding='auto',
                                       opt_method='Nelder-Mead',
                                       maxiter=maxiter)
            return {"params": params, "value": float(val), "tvgamma": tvgamma}
        except Exception:
            return None
