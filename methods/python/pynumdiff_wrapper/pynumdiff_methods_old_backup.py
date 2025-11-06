"""
PyNumDiff-Based Methods for Derivative Estimation

This module implements derivative estimation methods using the PyNumDiff package.
PyNumDiff provides numerical differentiation of noisy time-series data with automatic
parameter selection based on Holoborodko (2008) and other classical methods.

Methods organized by family:
1. Smooth-then-differentiate (smooth signal first, then compute all derivative orders):
   - Butterworth filtering
   - Spline smoothing
   - Gaussian kernel smoothing
   - Friedrichs mollification

2. Kalman smoothing (orders 0-2 only):
   - RTS smoother with state-space models

3. Total Variation regularization (order-specific):
   - TV velocity (1st derivative)
   - TV acceleration (2nd derivative)
   - TV jerk (3rd derivative)
   - Iterative TV solver

Reference:
- PyNumDiff documentation: https://github.com/florisvb/PyNumDiff
- JOSS paper: Djerroud et al. (2022)
"""

from typing import Dict, Optional
import numpy as np
import warnings

from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks

# Import PyNumDiff methods
try:
    import pynumdiff
    from pynumdiff import smooth_finite_difference as sfd
    from pynumdiff import total_variation_regularization as tvr
    from pynumdiff import kalman_smooth
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
                Smooth-then-differentiate (orders 0-7):
                - "PyNumDiff-Butter-Auto": Butterworth with automatic parameters
                - "PyNumDiff-Butter-Tuned": Butterworth with tuned parameters
                - "PyNumDiff-Spline-Auto": Spline smoothing with automatic parameters
                - "PyNumDiff-Spline-Tuned": Spline smoothing with tuned parameters
                - "PyNumDiff-Gaussian-Auto": Gaussian kernel with automatic parameters
                - "PyNumDiff-Gaussian-Tuned": Gaussian kernel with tuned parameters
                - "PyNumDiff-Friedrichs-Auto": Friedrichs with automatic parameters
                - "PyNumDiff-Friedrichs-Tuned": Friedrichs with tuned parameters

                Kalman smoothing (orders 0-2 only):
                - "PyNumDiff-Kalman-Auto": Kalman RTS with automatic parameters
                - "PyNumDiff-Kalman-Tuned": Kalman RTS with tuned parameters

                Total Variation regularization (order-specific):
                - "PyNumDiff-TV-Velocity": TV for 1st derivative only
                - "PyNumDiff-TV-Acceleration": TV for 2nd derivative only
                - "PyNumDiff-TV-Jerk": TV for 3rd derivative only
                - "PyNumDiff-TV-Iterative": Iterative TV for 1st derivative

        Returns:
            Dictionary with predictions, failures, and metadata
        """
        if not PYNUMDIFF_AVAILABLE:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": "PyNumDiff package not available"}
            }

        # Smooth-then-differentiate methods
        if method_name == "PyNumDiff-Butter-Auto":
            return self._butterdiff(regime="auto")
        elif method_name == "PyNumDiff-Butter-Tuned":
            return self._butterdiff(regime="tuned")
        elif method_name == "PyNumDiff-Spline-Auto":
            return self._splinediff(regime="auto")
        elif method_name == "PyNumDiff-Spline-Tuned":
            return self._splinediff(regime="tuned")
        elif method_name == "PyNumDiff-Gaussian-Auto":
            return self._gaussiandiff(regime="auto")
        elif method_name == "PyNumDiff-Gaussian-Tuned":
            return self._gaussiandiff(regime="tuned")
        elif method_name == "PyNumDiff-Friedrichs-Auto":
            return self._friedrichsdiff(regime="auto")
        elif method_name == "PyNumDiff-Friedrichs-Tuned":
            return self._friedrichsdiff(regime="tuned")

        # Kalman smoothing methods (orders 0-2 only)
        elif method_name == "PyNumDiff-Kalman-Auto":
            return self._kalman_smooth(regime="auto")
        elif method_name == "PyNumDiff-Kalman-Tuned":
            return self._kalman_smooth(regime="tuned")

        # Total Variation regularization (order-specific)
        elif method_name == "PyNumDiff-TV-Velocity":
            return self._tv_velocity()
        elif method_name == "PyNumDiff-TV-Acceleration":
            return self._tv_acceleration()
        elif method_name == "PyNumDiff-TV-Jerk":
            return self._tv_jerk()
        elif method_name == "PyNumDiff-TV-Iterative":
            return self._tv_iterative_velocity()

        else:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": f"Unknown PyNumDiff method: {method_name}"}
            }

    def _estimate_cutoff_frequency(self, y: np.ndarray, dt: float) -> float:
        """
        Estimate cutoff frequency for automatic parameter selection.

        Uses the PyNumDiff formula: gamma = exp(-1.6*log(f_cutoff) - 0.71*log(dt) - 5.1)
        We can invert this to estimate f_cutoff from signal characteristics.

        For simplicity, we use a heuristic based on signal spectrum.

        Args:
            y: Signal values
            dt: Time step

        Returns:
            Estimated cutoff frequency
        """
        # Try peak counting heuristic first (robust for approximately periodic signals)
        try:
            t_total = float(self.x_train[-1] - self.x_train[0]) if len(self.x_train) > 1 else (len(y) - 1) * dt
            if t_total > 0:
                # Use prominence relative to signal scale to avoid noise peaks
                scale = np.std(y) if np.std(y) > 0 else (np.max(y) - np.min(y) + 1e-9)
                peaks, _ = find_peaks(y, prominence=max(1e-3 * scale, 1e-9))
                peak_freq = float(len(peaks)) / t_total if t_total > 0 else 0.0
                if peak_freq > 0:
                    f_max = 0.5 / dt
                    peak_freq = max(0.01 * f_max, min(peak_freq, 0.3 * f_max))
                    return peak_freq
        except Exception:
            pass

        # Fallback: use FFT to estimate dominant frequency
        n = len(y)
        if n < 10:
            return 0.1  # Default fallback

        # Compute FFT
        yf = np.fft.fft(y - np.mean(y))
        freqs = np.fft.fftfreq(n, dt)
        power = np.abs(yf)**2

        # Find frequency with 90% cumulative power
        sorted_indices = np.argsort(power)[::-1]
        cumsum = np.cumsum(power[sorted_indices])
        idx_90 = np.searchsorted(cumsum, 0.9 * cumsum[-1])
        f_cutoff = np.abs(freqs[sorted_indices[idx_90]])

        # Clamp to reasonable range
        f_max = 0.5 / dt  # Nyquist frequency
        f_cutoff = max(0.01 * f_max, min(f_cutoff, 0.3 * f_max))

        return f_cutoff

    def _compute_tvgamma_from_cutoff(self, cutoff_frequency: float, dt: float) -> float:
        """
        Compute tvgamma using PyNumDiff's documented heuristic:
        tvgamma = exp(-1.6*log(cutoff_frequency) - 0.71*log(dt) - 5.1)
        """
        try:
            log_gamma = -1.6 * np.log(cutoff_frequency) - 0.71 * np.log(dt) - 5.1
            tvgamma = float(np.exp(log_gamma))
            return max(1e-8, min(tvgamma, 1e6))
        except Exception:
            return 1e-2

    def _tvgamma_docs_heuristic(self, y: np.ndarray, dt: float) -> float:
        """
        Estimate tvgamma via docs heuristic by first estimating cutoff frequency.
        """
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

    def _auto_tvgamma(self, y: np.ndarray, dt: float) -> float:
        """
        Automatic TV gamma parameter selection.

        Based on noise estimation and signal characteristics.

        Args:
            y: Signal values
            dt: Time step

        Returns:
            TV gamma parameter
        """
        # Estimate noise using MAD of first differences
        dy = np.diff(y)
        sigma_hat = np.median(np.abs(dy - np.median(dy))) / (0.6745 * np.sqrt(2.0))

        # TV gamma heuristic: gamma ∝ 1/σ
        gamma = 1.0 / max(sigma_hat, 1e-10)

        # Clamp to reasonable range
        gamma = max(1e-2, min(gamma, 1e4))

        return gamma

    def _smooth_and_differentiate_all_orders(self, x: np.ndarray, y_smooth: np.ndarray) -> Dict:
        """
        Helper: compute all derivative orders from a smoothed signal using splines.

        This is the key insight for smooth-then-differentiate methods:
        1. Use PyNumDiff to smooth the signal
        2. Fit a quintic spline to the smoothed signal
        3. Differentiate the spline to get all orders 0-7

        Args:
            x: Input grid points
            y_smooth: Smoothed signal values

        Returns:
            Dictionary with predictions and failures for all orders
        """
        predictions = {}
        failures = {}

        # UnivariateSpline is limited to k <= 5
        # For orders 0-5, use quintic spline
        # For orders 6-7, need make_interp_spline with higher k
        from scipy.interpolate import make_interp_spline

        max_order = max(self.orders) if self.orders else 0
        n_points = len(x)

        # Choose spline type and order based on max derivative needed
        if max_order <= 5:
            # Use UnivariateSpline (smoother, k=5)
            try:
                spline = UnivariateSpline(x, y_smooth, k=5, s=0.0)
            except Exception as e:
                # Fallback to cubic if quintic fails
                try:
                    spline = UnivariateSpline(x, y_smooth, k=3, s=0.0)
                except Exception as ee:
                    for order in self.orders:
                        failures[order] = f"spline_fit_failed: {ee}"
                        predictions[order] = [np.nan] * len(self.x_eval)
                    return {"predictions": predictions, "failures": failures}
        else:
            # Need orders 6-7: use make_interp_spline with higher k
            k = min(max_order + 2, n_points - 1)
            if k % 2 == 0:
                k = k - 1  # Make k odd
            try:
                spline = make_interp_spline(x, y_smooth, k=k)
            except Exception as e:
                # Fallback to k=5 make_interp_spline
                try:
                    spline = make_interp_spline(x, y_smooth, k=5)
                except Exception as ee:
                    for order in self.orders:
                        failures[order] = f"spline_fit_failed: {ee}"
                        predictions[order] = [np.nan] * len(self.x_eval)
                    return {"predictions": predictions, "failures": failures}

        # Compute derivatives at evaluation points
        for order in self.orders:
            try:
                if order == 0:
                    # Order 0: smoothed signal itself
                    vals = spline(self.x_eval)
                else:
                    # Higher orders: differentiate spline
                    # BSpline uses nu=, UnivariateSpline uses n=
                    try:
                        deriv_spline = spline.derivative(nu=order)
                    except TypeError:
                        deriv_spline = spline.derivative(n=order)
                    vals = deriv_spline(self.x_eval)

                predictions[order] = [float(v) for v in np.asarray(vals)]
            except Exception as e:
                failures[order] = str(e)
                predictions[order] = [np.nan] * len(self.x_eval)

        return {"predictions": predictions, "failures": failures}

    def _butterdiff(self, regime: str = "auto") -> Dict:
        """
        Butterworth filtering with smoothing and differentiation.

        Uses PyNumDiff's butterdiff for smoothing, then computes all derivative
        orders via spline differentiation.

        Args:
            regime: "auto" for automatic parameters or "tuned" for conservative values

        Returns:
            Dictionary with predictions, failures, and metadata
        """
        t = self.x_train
        y = self.y_train
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 1.0

        if regime == "auto":
            opt = self._optimize_params(sfd.butterdiff, y, dt)
            if opt is None:
                return {"predictions": {}, "failures": {"error": "butterdiff_optimize_failed"}}
            try:
                x_smooth, dx_dt = sfd.butterdiff(y, dt, **opt["params"])  # type: ignore[arg-type]
            except Exception as e:
                return {"predictions": {}, "failures": {"error": f"butterdiff_run_failed: {e}"}}
            result = self._smooth_and_differentiate_all_orders(t, x_smooth)
            meta = {"method": "butterdiff", "regime": regime, "dt": dt,
                    "opt_params": opt["params"], "opt_value": opt["value"], "tvgamma": opt["tvgamma"]}
            return {"predictions": result["predictions"], "failures": result["failures"], "meta": meta}
        else:  # tuned
            # Conservative parameters for robustness
            cutoff_norm = 0.2  # 20% of Nyquist

        try:
            # Call PyNumDiff butterdiff (returns smoothed signal and 1st derivative)
            # API: butterdiff(x, dt, filter_order=2, cutoff_freq=0.5)
            x_smooth, dx_dt = sfd.butterdiff(y, dt, filter_order=2, cutoff_freq=cutoff_norm)

            # Compute all derivative orders from smoothed signal
            result = self._smooth_and_differentiate_all_orders(t, x_smooth)

            meta = {
                "method": "butterdiff",
                "regime": regime,
                "dt": dt,
                "cutoff_freq": cutoff_norm
            }

            return {
                "predictions": result["predictions"],
                "failures": result["failures"],
                "meta": meta
            }

        except Exception as e:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": str(e)}
            }

    def _splinediff(self, regime: str = "auto") -> Dict:
        """
        Spline smoothing with differentiation.

        Uses PyNumDiff's splinediff for smoothing, then computes all derivative
        orders via spline differentiation.

        Args:
            regime: "auto" for automatic parameters or "tuned" for conservative values

        Returns:
            Dictionary with predictions, failures, and metadata
        """
        t = self.x_train
        y = self.y_train
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 1.0

        try:
            if regime == "auto":
                opt = self._optimize_params(sfd.splinediff, y, dt)
                if opt is None:
                    return {"predictions": {}, "failures": {"error": "splinediff_optimize_failed"}}
                x_smooth, dx_dt = sfd.splinediff(y, dt, **opt["params"])  # type: ignore[arg-type]
            else:
                x_smooth, dx_dt = sfd.splinediff(y, dt)

            # Compute all derivative orders from smoothed signal
            result = self._smooth_and_differentiate_all_orders(t, x_smooth)

            meta = {"method": "splinediff", "regime": regime, "dt": dt}
            if regime == "auto":
                meta.update({"opt_params": opt["params"], "opt_value": opt["value"], "tvgamma": opt["tvgamma"]})

            return {
                "predictions": result["predictions"],
                "failures": result["failures"],
                "meta": meta
            }

        except Exception as e:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": str(e)}
            }

    def _gaussiandiff(self, regime: str = "auto") -> Dict:
        """
        Gaussian kernel smoothing with differentiation.

        Uses PyNumDiff's gaussiandiff for smoothing, then computes all derivative
        orders via spline differentiation.

        Args:
            regime: "auto" for automatic parameters or "tuned" for conservative values

        Returns:
            Dictionary with predictions, failures, and metadata
        """
        t = self.x_train
        y = self.y_train
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 1.0

        if regime == "auto":
            opt = self._optimize_params(sfd.gaussiandiff, y, dt)
            if opt is None:
                return {"predictions": {}, "failures": {"error": "gaussiandiff_optimize_failed"}}
            try:
                x_smooth, dx_dt = sfd.gaussiandiff(y, dt, **opt["params"])  # type: ignore[arg-type]
            except Exception as e:
                return {"predictions": {}, "failures": {"error": f"gaussiandiff_run_failed: {e}"}}
            result = self._smooth_and_differentiate_all_orders(t, x_smooth)
            meta = {"method": "gaussiandiff", "regime": regime, "dt": dt,
                    "opt_params": opt["params"], "opt_value": opt["value"], "tvgamma": opt["tvgamma"]}
            return {"predictions": result["predictions"], "failures": result["failures"], "meta": meta}
        else:  # tuned
            window = min(11, n if n % 2 == 1 else n - 1)

        try:
            # Call PyNumDiff gaussiandiff
            # API: gaussiandiff(x, dt, window_size=5)
            x_smooth, dx_dt = sfd.gaussiandiff(y, dt, window_size=window)

            # Compute all derivative orders from smoothed signal
            result = self._smooth_and_differentiate_all_orders(t, x_smooth)

            meta = {"method": "gaussiandiff", "regime": regime, "dt": dt, "window_size": window}

            return {
                "predictions": result["predictions"],
                "failures": result["failures"],
                "meta": meta
            }

        except Exception as e:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": str(e)}
            }

    def _friedrichsdiff(self, regime: str = "auto") -> Dict:
        """
        Friedrichs mollification with differentiation.

        Uses PyNumDiff's friedrichsdiff for smoothing, then computes all derivative
        orders via spline differentiation.

        Args:
            regime: "auto" for automatic parameters or "tuned" for conservative values

        Returns:
            Dictionary with predictions, failures, and metadata
        """
        t = self.x_train
        y = self.y_train
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 1.0

        if regime == "auto":
            opt = self._optimize_params(sfd.friedrichsdiff, y, dt)
            if opt is None:
                return {"predictions": {}, "failures": {"error": "friedrichsdiff_optimize_failed"}}
            try:
                x_smooth, dx_dt = sfd.friedrichsdiff(y, dt, **opt["params"])  # type: ignore[arg-type]
            except Exception as e:
                return {"predictions": {}, "failures": {"error": f"friedrichsdiff_run_failed: {e}"}}
            result = self._smooth_and_differentiate_all_orders(t, x_smooth)
            meta = {"method": "friedrichsdiff", "regime": regime, "dt": dt,
                    "opt_params": opt["params"], "opt_value": opt["value"], "tvgamma": opt["tvgamma"]}
            return {"predictions": result["predictions"], "failures": result["failures"], "meta": meta}
        else:  # tuned
            window = min(11, n if n % 2 == 1 else n - 1)

        try:
            # Call PyNumDiff friedrichsdiff
            # API: friedrichsdiff(x, dt, window_size=5)
            x_smooth, dx_dt = sfd.friedrichsdiff(y, dt, window_size=window)

            # Compute all derivative orders from smoothed signal
            result = self._smooth_and_differentiate_all_orders(t, x_smooth)

            meta = {"method": "friedrichsdiff", "regime": regime, "dt": dt, "window_size": window}

            return {
                "predictions": result["predictions"],
                "failures": result["failures"],
                "meta": meta
            }

        except Exception as e:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": str(e)}
            }

    def _kalman_smooth(self, regime: str = "auto") -> Dict:
        """
        Kalman RTS smoother for derivative estimation (orders 0-2 only).

        Uses PyNumDiff's kalman_smooth which returns position, velocity, and
        acceleration estimates. Higher orders computed via spline if requested.

        Args:
            regime: "auto" for automatic parameters or "tuned" for conservative values

        Returns:
            Dictionary with predictions, failures, and metadata
        """
        t = self.x_train
        y = self.y_train
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 1.0

        if regime == "auto":
            opt = self._optimize_params(kalman_smooth.rtsdiff, y, dt, maxiter=30)
            if opt is None:
                return {"predictions": {}, "failures": {"error": "kalman_rtsdiff_optimize_failed"}}
            try:
                x_smooth, dx_smooth = kalman_smooth.rtsdiff(y, dt, **opt["params"])  # type: ignore[arg-type]
            except Exception as e:
                return {"predictions": {}, "failures": {"error": f"kalman_rtsdiff_run_failed: {e}"}}

            predictions = {}
            failures = {}

            for order in self.orders:
                try:
                    if order == 0:
                        spline = UnivariateSpline(t, x_smooth, k=3, s=0.0)
                        predictions[order] = [float(v) for v in spline(self.x_eval)]
                    elif order == 1:
                        spline = UnivariateSpline(t, dx_smooth, k=3, s=0.0)
                        predictions[order] = [float(v) for v in spline(self.x_eval)]
                    else:
                        from scipy.interpolate import make_interp_spline
                        if order <= 5:
                            spline = UnivariateSpline(t, x_smooth, k=5, s=0.0)
                            deriv_spline = spline.derivative(n=order)
                        else:
                            k = min(order + 2, len(t) - 1)
                            if k % 2 == 0:
                                k = k - 1
                            spline = make_interp_spline(t, x_smooth, k=k)
                            deriv_spline = spline.derivative(nu=order)
                        predictions[order] = [float(v) for v in deriv_spline(self.x_eval)]
                except Exception as e:
                    failures[order] = str(e)
                    predictions[order] = [np.nan] * len(self.x_eval)

            meta = {"method": "kalman_smooth/rtsdiff", "regime": regime, "dt": dt,
                    "opt_params": opt["params"], "opt_value": opt["value"], "tvgamma": opt["tvgamma"]}
            return {"predictions": predictions, "failures": failures, "meta": meta}
        else:  # tuned
            process_var = 1e-6
            measurement_var = 1e-8

        try:
            # Call PyNumDiff kalman_smooth
            # API: constant_acceleration(x, dt, r=measurement_noise, q=process_noise)
            # Returns: x_smooth (position), dx_smooth (velocity) - only 2 values in v0.2!
            x_smooth, dx_smooth = kalman_smooth.constant_acceleration(
                y, dt, r=measurement_var, q=process_var
            )

            predictions = {}
            failures = {}

            # For orders 0-1, use Kalman outputs directly; for 2+, differentiate via spline
            for order in self.orders:
                try:
                    if order == 0:
                        # Position: interpolate to x_eval
                        spline = UnivariateSpline(t, x_smooth, k=3, s=0.0)
                        predictions[order] = [float(v) for v in spline(self.x_eval)]
                    elif order == 1:
                        # Velocity: interpolate to x_eval
                        spline = UnivariateSpline(t, dx_smooth, k=3, s=0.0)
                        predictions[order] = [float(v) for v in spline(self.x_eval)]
                    else:
                        # Order 2+: differentiate position spline
                        # UnivariateSpline limited to k <= 5, use make_interp_spline for higher orders
                        from scipy.interpolate import make_interp_spline

                        if order <= 5:
                            spline = UnivariateSpline(t, x_smooth, k=5, s=0.0)
                            deriv_spline = spline.derivative(n=order)
                        else:
                            k = min(order + 2, len(t) - 1)
                            if k % 2 == 0:
                                k = k - 1
                            spline = make_interp_spline(t, x_smooth, k=k)
                            deriv_spline = spline.derivative(nu=order)

                        predictions[order] = [float(v) for v in deriv_spline(self.x_eval)]
                except Exception as e:
                    failures[order] = str(e)
                    predictions[order] = [np.nan] * len(self.x_eval)

            meta = {
                "method": "kalman_smooth",
                "regime": regime,
                "dt": dt,
                "r": measurement_var,
                "q": process_var
            }

            return {
                "predictions": predictions,
                "failures": failures,
                "meta": meta
            }

        except Exception as e:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": str(e)}
            }

    def _tv_velocity(self) -> Dict:
        """
        Total Variation regularization for 1st derivative (velocity).

        Uses PyNumDiff's TV regularization specifically for velocity estimation.

        Returns:
            Dictionary with predictions, failures, and metadata
        """
        t = self.x_train
        y = self.y_train
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 1.0

        try:
            opt = self._optimize_params(tvr.velocity, y, dt)
            if opt is None:
                return {"predictions": {}, "failures": {"error": "tv_velocity_optimize_failed"}}
            x_smooth, dx_dt = tvr.velocity(y, dt, **opt["params"])  # type: ignore[arg-type]

            predictions = {}
            failures = {}

            # Interpolate velocity to x_eval
            # For higher orders, we need (order-1)th derivative of velocity
            # For order 7, need 6th derivative, so need k >= 6
            from scipy.interpolate import make_interp_spline
            max_vel_deriv = max([o - 1 for o in self.orders if o > 1], default=0)

            if max_vel_deriv <= 3:
                spline = UnivariateSpline(t, dx_dt, k=3, s=0.0)
            elif max_vel_deriv <= 5:
                spline = UnivariateSpline(t, dx_dt, k=5, s=0.0)
            else:
                k = min(max_vel_deriv + 2, len(t) - 1)
                if k % 2 == 0:
                    k = k - 1
                spline = make_interp_spline(t, dx_dt, k=k)

            for order in self.orders:
                try:
                    if order == 0:
                        # Order 0: integrate velocity (or use original signal via spline)
                        spline0 = UnivariateSpline(t, y, k=3, s=0.0)
                        predictions[order] = [float(v) for v in spline0(self.x_eval)]
                    elif order == 1:
                        # Order 1: velocity from TV
                        predictions[order] = [float(v) for v in spline(self.x_eval)]
                    else:
                        # Higher orders: differentiate velocity spline
                        # Handle both UnivariateSpline (n=) and BSpline (nu=)
                        try:
                            deriv_spline = spline.derivative(nu=order-1)
                        except TypeError:
                            deriv_spline = spline.derivative(n=order-1)
                        predictions[order] = [float(v) for v in deriv_spline(self.x_eval)]
                except Exception as e:
                    failures[order] = str(e)
                    predictions[order] = [np.nan] * len(self.x_eval)

            meta = {"method": "tv_velocity", "dt": dt, "opt_params": opt["params"], "opt_value": opt["value"], "tvgamma": opt["tvgamma"]}

            return {
                "predictions": predictions,
                "failures": failures,
                "meta": meta
            }

        except Exception as e:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": str(e)}
            }

    def _tv_acceleration(self) -> Dict:
        """
        Total Variation regularization for 2nd derivative (acceleration).

        Uses PyNumDiff's TV regularization specifically for acceleration estimation.

        Returns:
            Dictionary with predictions, failures, and metadata
        """
        t = self.x_train
        y = self.y_train
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 1.0

        try:
            opt = self._optimize_params(tvr.acceleration, y, dt)
            if opt is None:
                return {"predictions": {}, "failures": {"error": "tv_acceleration_optimize_failed"}}
            x_smooth, ddx_dt = tvr.acceleration(y, dt, **opt["params"])  # type: ignore[arg-type]

            predictions = {}
            failures = {}

            # Interpolate acceleration to x_eval
            # For order 3+, need (order-2)th derivative of acceleration
            # For order 7, need 5th derivative, so need k >= 5
            from scipy.interpolate import make_interp_spline
            max_accel_deriv = max([o - 2 for o in self.orders if o > 2], default=0)

            if max_accel_deriv <= 3:
                spline = UnivariateSpline(t, ddx_dt, k=3, s=0.0)
            elif max_accel_deriv <= 5:
                spline = UnivariateSpline(t, ddx_dt, k=5, s=0.0)
            else:
                k = min(max_accel_deriv + 2, len(t) - 1)
                if k % 2 == 0:
                    k = k - 1
                spline = make_interp_spline(t, ddx_dt, k=k)

            for order in self.orders:
                try:
                    if order == 0:
                        # Order 0: use original signal via spline
                        spline0 = UnivariateSpline(t, y, k=3, s=0.0)
                        predictions[order] = [float(v) for v in spline0(self.x_eval)]
                    elif order == 1:
                        # Order 1: integrate acceleration (or differentiate order 0)
                        spline0 = UnivariateSpline(t, y, k=3, s=0.0)
                        deriv1 = spline0.derivative(n=1)
                        predictions[order] = [float(v) for v in deriv1(self.x_eval)]
                    elif order == 2:
                        # Order 2: acceleration from TV
                        predictions[order] = [float(v) for v in spline(self.x_eval)]
                    else:
                        # Higher orders: differentiate acceleration spline
                        # Handle both UnivariateSpline (n=) and BSpline (nu=)
                        try:
                            deriv_spline = spline.derivative(nu=order-2)
                        except TypeError:
                            deriv_spline = spline.derivative(n=order-2)
                        predictions[order] = [float(v) for v in deriv_spline(self.x_eval)]
                except Exception as e:
                    failures[order] = str(e)
                    predictions[order] = [np.nan] * len(self.x_eval)

            meta = {"method": "tv_acceleration", "dt": dt, "opt_params": opt["params"], "opt_value": opt["value"], "tvgamma": opt["tvgamma"]}

            return {
                "predictions": predictions,
                "failures": failures,
                "meta": meta
            }

        except Exception as e:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": str(e)}
            }

    def _tv_jerk(self) -> Dict:
        """
        Total Variation regularization for 3rd derivative (jerk).

        Uses PyNumDiff's TV regularization specifically for jerk estimation.

        Returns:
            Dictionary with predictions, failures, and metadata
        """
        t = self.x_train
        y = self.y_train
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 1.0

        try:
            opt = self._optimize_params(tvr.jerk, y, dt)
            if opt is None:
                return {"predictions": {}, "failures": {"error": "tv_jerk_optimize_failed"}}
            x_smooth, dddx_dt = tvr.jerk(y, dt, **opt["params"])  # type: ignore[arg-type]

            predictions = {}
            failures = {}

            # Interpolate jerk to x_eval
            # For order 7, we need 4th derivative of jerk, so k must be >= 4
            # Use k=5 if any order > 6 is requested, otherwise k=3 is fine
            max_jerk_deriv = max([o - 3 for o in self.orders if o > 3], default=0)
            k_jerk = 5 if max_jerk_deriv > 3 else 3
            spline = UnivariateSpline(t, dddx_dt, k=k_jerk, s=0.0)

            for order in self.orders:
                try:
                    if order < 3:
                        # Orders 0-2: use original signal via spline (k=5 sufficient)
                        spline0 = UnivariateSpline(t, y, k=5, s=0.0)
                        if order == 0:
                            predictions[order] = [float(v) for v in spline0(self.x_eval)]
                        else:
                            deriv = spline0.derivative(n=order)
                            predictions[order] = [float(v) for v in deriv(self.x_eval)]
                    elif order == 3:
                        # Order 3: jerk from TV
                        predictions[order] = [float(v) for v in spline(self.x_eval)]
                    else:
                        # Higher orders: differentiate jerk spline
                        deriv_spline = spline.derivative(n=order-3)
                        predictions[order] = [float(v) for v in deriv_spline(self.x_eval)]
                except Exception as e:
                    failures[order] = str(e)
                    predictions[order] = [np.nan] * len(self.x_eval)

            meta = {"method": "tv_jerk", "dt": dt, "opt_params": opt.get("params", {}), "opt_value": opt.get("value", None), "tvgamma": opt.get("tvgamma", None)}

            return {
                "predictions": predictions,
                "failures": failures,
                "meta": meta
            }

        except Exception as e:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": str(e)}
            }

    def _tv_iterative_velocity(self) -> Dict:
        """
        Iterative Total Variation regularization for 1st derivative (velocity).

        Uses PyNumDiff's iterative TV solver for velocity estimation with
        automatic parameter selection.

        Returns:
            Dictionary with predictions, failures, and metadata
        """
        t = self.x_train
        y = self.y_train
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 1.0

        try:
            opt = self._optimize_params(tvr.iterative_velocity, y, dt)
            if opt is None:
                return {"predictions": {}, "failures": {"error": "tv_iterative_velocity_optimize_failed"}}
            x_smooth, dx_dt = tvr.iterative_velocity(y, dt, **opt["params"])  # type: ignore[arg-type]

            predictions = {}
            failures = {}

            # Interpolate velocity to x_eval
            # For order 2+, need (order-1)th derivative of velocity
            # For order 7, need 6th derivative, so need k >= 6
            from scipy.interpolate import make_interp_spline
            max_vel_deriv = max([o - 1 for o in self.orders if o > 1], default=0)

            if max_vel_deriv <= 3:
                spline = UnivariateSpline(t, dx_dt, k=3, s=0.0)
            elif max_vel_deriv <= 5:
                spline = UnivariateSpline(t, dx_dt, k=5, s=0.0)
            else:
                k = min(max_vel_deriv + 2, len(t) - 1)
                if k % 2 == 0:
                    k = k - 1
                spline = make_interp_spline(t, dx_dt, k=k)

            for order in self.orders:
                try:
                    if order == 0:
                        # Order 0: use original signal via spline
                        spline0 = UnivariateSpline(t, y, k=3, s=0.0)
                        predictions[order] = [float(v) for v in spline0(self.x_eval)]
                    elif order == 1:
                        # Order 1: velocity from iterative TV
                        predictions[order] = [float(v) for v in spline(self.x_eval)]
                    else:
                        # Higher orders: differentiate velocity spline
                        # Handle both UnivariateSpline (n=) and BSpline (nu=)
                        try:
                            deriv_spline = spline.derivative(nu=order-1)
                        except TypeError:
                            deriv_spline = spline.derivative(n=order-1)
                        predictions[order] = [float(v) for v in deriv_spline(self.x_eval)]
                except Exception as e:
                    failures[order] = str(e)
                    predictions[order] = [np.nan] * len(self.x_eval)

            meta = {"method": "tv_iterative_velocity", "dt": dt, "iterations": 50, "opt_params": opt["params"], "opt_value": opt["value"], "tvgamma": opt["tvgamma"]}

            return {
                "predictions": predictions,
                "failures": failures,
                "meta": meta
            }

        except Exception as e:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": str(e)}
            }
