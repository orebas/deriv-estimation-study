"""
Spline-Based Methods for Derivative Estimation

This module implements spline-based derivative estimation methods including:
1. Chebyshev polynomial approximation (fixed and adaptive degree selection)
2. RKHS smoothing splines (m=2 Sobolev penalty)
3. Butterworth lowpass filtering + quintic spline derivatives
4. Finite difference baseline with smoothing + spline
5. SVR regression + quintic spline derivatives
"""

from typing import Dict
import numpy as np
import os

from numpy.polynomial.chebyshev import Chebyshev
from scipy.interpolate import UnivariateSpline
from scipy.signal import butter, filtfilt
from sklearn.svm import SVR

# Import base class from common utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import MethodEvaluator, ADAPTIVE_HYPERPARAMS

if ADAPTIVE_HYPERPARAMS:
    from hyperparameters import select_chebyshev_degree


class SplineMethods(MethodEvaluator):
    """
    Spline-based methods for derivative estimation.

    Inherits from base MethodEvaluator and implements spline-specific methods.
    """

    def evaluate_method(self, method_name: str) -> Dict:
        """
        Dispatch to the appropriate spline method based on method name.

        Args:
            method_name: Name of method to evaluate. Supported:
                - "Chebyshev-Basic-Python": Global Chebyshev polynomial (fixed degree)
                - "Chebyshev-AICc": Chebyshev with adaptive degree selection
                - "RKHS_Spline_m2_Python": RKHS smoothing spline (m=2)
                - "Butterworth_Python": Butterworth lowpass + quintic spline
                - "ButterworthSpline_Python": Finite diff + smoothing + spline
                - "SVR_Python": SVR regression + quintic spline

        Returns:
            Dictionary with predictions, failures, and metadata
        """
        if method_name == "Chebyshev-Basic-Python":
            return self._chebyshev()
        elif method_name == "Chebyshev-AICc":
            return self._chebyshev_aicc()
        elif method_name == "RKHS_Spline_m2_Python":
            return self._rkhs_spline_m2()
        elif method_name == "Butterworth_Python":
            return self._butterworth_spline()
        elif method_name == "ButterworthSpline_Python":
            return self._finite_diff_spline()
        elif method_name == "SVR_Python":
            return self._svr_spline()
        else:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": f"Unknown spline method: {method_name}"}
            }

    def _chebyshev(self) -> Dict:
        """Global Chebyshev polynomial with analytic derivatives.

        Uses fixed degree selection based on training set size.
        For adaptive selection, use Chebyshev-AICc method instead.

        Returns:
            Dictionary containing:
                predictions: Dict mapping derivative order to list of predictions
                failures: Dict of any errors encountered
                meta: Polynomial degree and selection method
        """
        tmin = float(self.x_train.min())
        tmax = float(self.x_train.max())
        n_train = len(self.x_train)

        # Use fixed degree (original behavior)
        # For adaptive selection, use Chebyshev-Adaptive method instead
        deg = max(3, min(20, n_train - 1))
        aicc = None
        selection_method = "fixed"

        poly = Chebyshev.fit(self.x_train, self.y_train, deg=deg, domain=[tmin, tmax])

        predictions = {}
        failures = {}
        for order in self.orders:
            try:
                if order == 0:
                    vals = poly(self.x_eval)
                else:
                    dpoly = poly.deriv(m=order)
                    vals = dpoly(self.x_eval)
                predictions[order] = [float(v) for v in np.asarray(vals)]
            except Exception as e:
                failures[order] = str(e)
                predictions[order] = [np.nan] * len(self.x_eval)

        meta = {"degree": deg, "selection": selection_method}
        if aicc is not None:
            meta["aicc"] = float(aicc)

        return {"predictions": predictions, "failures": failures, "meta": meta}

    def _chebyshev_aicc(self) -> Dict:
        """
        Chebyshev polynomial with adaptive degree selection via AICc.

        Uses Akaike Information Criterion (corrected for small samples) to
        automatically select polynomial degree based on bias-variance tradeoff.

        Returns:
            Dictionary containing:
                predictions: Dict mapping derivative order to list of predictions
                failures: Dict of any errors encountered
                meta: Polynomial degree, AICc score, and selection method
        """
        if not ADAPTIVE_HYPERPARAMS:
            raise RuntimeError("hyperparameters module not available")

        tmin = float(self.x_train.min())
        tmax = float(self.x_train.max())
        n_train = len(self.x_train)

        # Always use adaptive degree selection via AICc
        deg, aicc = select_chebyshev_degree(
            self.x_train, self.y_train,
            max_degree=min(30, n_train - 1),
            min_degree=3
        )

        poly = Chebyshev.fit(self.x_train, self.y_train, deg=deg, domain=[tmin, tmax])

        predictions = {}
        failures = {}
        for order in self.orders:
            try:
                if order == 0:
                    vals = poly(self.x_eval)
                else:
                    dpoly = poly.deriv(m=order)
                    vals = dpoly(self.x_eval)
                predictions[order] = [float(v) for v in np.asarray(vals)]
            except Exception as e:
                failures[order] = str(e)
                predictions[order] = [np.nan] * len(self.x_eval)

        meta = {"degree": deg, "selection": "AICc", "aicc": float(aicc)}
        return {"predictions": predictions, "failures": failures, "meta": meta}

    def _quintic_spline_derivatives(self, x_grid: np.ndarray, y_grid: np.ndarray) -> Dict:
        """Helper: fit quintic spline to (x_grid, y_grid) and return derivatives at self.x_eval.

        This is an internal helper method used by other spline methods to compute derivatives
        from a smoothed or filtered signal.

        Args:
            x_grid: Input grid points
            y_grid: Function values at grid points

        Returns:
            Dictionary containing predictions and failures for all derivative orders
        """
        predictions = {}
        failures = {}
        try:
            us = UnivariateSpline(x_grid, y_grid, k=5, s=0.0)
        except Exception as e:
            # Fall back to cubic if quintic fails
            try:
                us = UnivariateSpline(x_grid, y_grid, k=3, s=0.0)
            except Exception as ee:
                for order in self.orders:
                    failures[order] = f"spline_fit_failed: {ee}"
                    predictions[order] = [np.nan] * len(self.x_eval)
                return {"predictions": predictions, "failures": failures}

        for order in self.orders:
            try:
                d = us.derivative(n=order)
                vals = d(self.x_eval)
                predictions[order] = [float(v) for v in np.asarray(vals)]
            except Exception as e:
                failures[order] = str(e)
                predictions[order] = [np.nan] * len(self.x_eval)
        return {"predictions": predictions, "failures": failures}

    def _rkhs_spline_m2(self) -> Dict:
        """Smoothing spline targeting Sobolev m=2 (penalize integral of f''^2).

        Implemented via UnivariateSpline with smoothing s selected from env or heuristic.
        Higher derivatives obtained via spline's derivative method.

        The smoothing parameter can be controlled via environment variables:
        - RKHS_M2_S: Explicit smoothing parameter value
        - RKHS_NOISE_FLOOR: Noise floor for heuristic (default 0.01)

        Returns:
            Dictionary containing:
                predictions: Dict mapping derivative order to list of predictions
                failures: Dict of any errors encountered
                meta: Smoothing parameter value
        """
        t = self.x_train
        y = self.y_train
        n = len(t)
        s_env = os.environ.get("RKHS_M2_S", "")
        if s_env:
            try:
                s = float(s_env)
            except Exception:
                s = None
        else:
            # Heuristic smoothing ~ n * (noise_floor * std(y))^2
            noise_floor = float(os.environ.get("RKHS_NOISE_FLOOR", "0.01"))
            s = n * (noise_floor * float(np.std(y) + 1e-12)) ** 2
        try:
            spl = UnivariateSpline(t, y, k=5, s=s)
        except Exception:
            spl = UnivariateSpline(t, y, k=3, s=s)

        predictions = {}
        failures = {}
        for order in self.orders:
            try:
                d = spl.derivative(n=order)
                vals = d(self.x_eval)
                predictions[order] = [float(v) for v in np.asarray(vals)]
            except Exception as e:
                failures[order] = str(e)
                predictions[order] = [np.nan] * len(self.x_eval)
        return {"predictions": predictions, "failures": failures, "meta": {"s": s}}

    def _butterworth_spline(self) -> Dict:
        """Butterworth lowpass smoothing + quintic spline derivatives.

        Applies a 4th order Butterworth lowpass filter at 0.1 Nyquist frequency
        to smooth the signal, then computes derivatives via quintic spline.

        Returns:
            Dictionary containing predictions and failures from quintic spline
        """
        t = self.x_train
        y = self.y_train
        n = len(t)
        if n < 5:
            raise ValueError("Not enough points for Butterworth smoothing")
        # Design a 4th order low-pass at 0.1 Nyquist
        b, a = butter(N=4, Wn=0.1, btype='lowpass', analog=False)
        y_smooth = filtfilt(b, a, y)
        return self._quintic_spline_derivatives(t, y_smooth)

    def _finite_diff_spline(self) -> Dict:
        """Legacy FiniteDiff baseline replaced by smoothing + quintic-spline derivatives.

        Uses a milder lowpass filter (2nd order Butterworth at 0.2 Nyquist)
        to mimic a pre-filter prior to differentiation.

        Returns:
            Dictionary containing predictions and failures from quintic spline
        """
        # Use a milder lowpass to mimic a pre-filter prior to differentiation
        t = self.x_train
        y = self.y_train
        b, a = butter(N=2, Wn=0.2, btype='lowpass', analog=False)
        y_smooth = filtfilt(b, a, y)
        return self._quintic_spline_derivatives(t, y_smooth)

    def _svr_spline(self) -> Dict:
        """SVR (RBF) fit, then quintic-spline derivatives of predicted signal.

        Fits a Support Vector Regression model with RBF kernel to the training data,
        evaluates at x_eval, then computes derivatives via quintic spline.

        Returns:
            Dictionary containing predictions and failures from quintic spline
        """
        X = self.x_train.reshape(-1, 1)
        y = self.y_train
        svr = SVR(kernel='rbf', C=10.0, gamma='scale', epsilon=0.01)
        svr.fit(X, y)
        yhat_eval = svr.predict(self.x_eval.reshape(-1, 1))
        return self._quintic_spline_derivatives(self.x_eval, yhat_eval)
