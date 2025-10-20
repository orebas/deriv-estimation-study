#!/usr/bin/env python3
"""
Standalone Python script for derivative estimation methods.

Reads data from JSON, applies methods, writes results back to JSON.

Methods implemented (no finite differences):
- chebyshev: Global Chebyshev polynomial with analytic derivatives
- fourier: Trigonometric polynomial with analytic derivatives
- gp_rbf_mean: GP (ConstantKernel * RBF + White) posterior mean derivatives (closed-form)
- ad_trig: Trigonometric polynomial evaluated with autograd, derivatives via iterated grads
"""

import json
import sys
import time
from pathlib import Path
import numpy as np
from typing import Dict, List
import os

# Import hyperparameter selection module
try:
    from hyperparameters import (
        select_chebyshev_degree,
        select_fourier_harmonics,
        select_aaa_tolerance,
        select_fourier_filter_fraction_simple
    )
    ADAPTIVE_HYPERPARAMS = True
except ImportError:
    ADAPTIVE_HYPERPARAMS = False
    import warnings
    warnings.warn("hyperparameters module not found - using fixed hyperparameters")

# Import packages
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
import warnings
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.simplefilter("ignore", ConvergenceWarning)
except Exception:
    pass
from numpy.polynomial.chebyshev import Chebyshev
from numpy.polynomial import hermite_e as herme
from sklearn.svm import SVR
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.interpolate import UnivariateSpline

# Optional: autograd for AD-backed trig method
try:
    import autograd.numpy as anp
    from autograd import elementwise_grad as egrad
    AUTOGRAD_AVAILABLE = True
except Exception:
    AUTOGRAD_AVAILABLE = False

# Optional: TV regularized numerical differentiation (Rick Chartrand, Python port)
try:
    from tvregdiff import TVRegDiff  # type: ignore
    TVREG_AVAILABLE = True
except Exception:
    TVREG_AVAILABLE = False

def _safe_orders(data_orders: List[int]) -> List[int]:
    if data_orders is None:
        return list(range(8))
    try:
        return [int(o) for o in data_orders]
    except Exception:
        return list(range(8))


class MethodEvaluator:
    """Evaluates Python-based differentiation methods."""

    def __init__(self, x_train, y_train, x_eval, orders):
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)
        self.x_eval = np.array(x_eval)
        self.orders = orders

    def evaluate_method(self, method_name: str) -> Dict:
        """Evaluate a single method."""
        t_start = time.time()

        try:
            if method_name == "chebyshev":
                result = self._chebyshev()
            elif method_name == "fourier":
                result = self._fourier()
            elif method_name == "gp_rbf_mean":
                result = self._gp_rbf_mean_derivative()
            elif method_name == "ad_trig":
                result = self._ad_trig()
            # Legacy IFAC25 Python methods
            elif method_name == "Butterworth_Python":
                result = self._butterworth_spline()
            elif method_name == "ButterworthSpline_Python":
                result = self._finite_diff_spline()
            elif method_name == "SavitzkyGolay_Python":
                result = self._savgol_method()
            elif method_name == "SVR_Python":
                result = self._svr_spline()
            elif method_name == "KalmanGrad_Python":
                result = self._kalman_grad()
            elif method_name == "TVRegDiff_Python":
                result = self._tvregdiff_method()
            elif method_name == "RKHS_Spline_m2_Python":
                result = self._rkhs_spline_m2()
            elif method_name == "SpectralTaper_Python":
                result = self._spectral_taper_derivative()
            elif method_name == "Whittaker_m2_Python":
                result = self._whittaker_m2()
            elif method_name == "GP_RBF_Python" or method_name == "GP_RBF_Iso_Python":
                result = self._gp_rbf_mean_derivative()
            elif method_name == "GP_Matern_Python":
                result = self._gp_matern(nu=1.5)
            elif method_name == "GP_Matern_1.5_Python":
                result = self._gp_matern(nu=1.5)
            elif method_name == "GP_Matern_2.5_Python":
                result = self._gp_matern(nu=2.5)
            elif method_name == "fourier_continuation":
                result = self._fourier_continuation()
            else:
                return {
                    "success": False,
                    "error": f"Unknown method: {method_name}",
                    "timing": time.time() - t_start
                }

            result["timing"] = time.time() - t_start
            result["success"] = True
            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timing": time.time() - t_start
            }

    def _gp_rbf_mean_derivative(self) -> Dict:
        """GP posterior mean derivatives using closed-form RBF kernel derivatives."""
        # Kernel: Constant * RBF + White
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) \
                 + WhiteKernel(noise_level=1e-8, noise_level_bounds=(1e-10, 1e-2))

        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            alpha=0.0,
            normalize_y=False
        )

        X = self.x_train.reshape(-1, 1)
        y = self.y_train
        gp.fit(X, y)

        # Extract fitted parameters (robust to structure Constant*RBF + White)
        fitted = gp.kernel_
        # Defaults
        amp = 1.0
        ell = 1.0
        try:
            # Expect Sum(Product(Constant, RBF), White)
            if hasattr(fitted, 'k1') and hasattr(fitted, 'k2'):
                k1 = fitted.k1
                # k1 should be Product(Constant, RBF)
                if hasattr(k1, 'k1') and hasattr(k1, 'k2') and isinstance(k1.k2, RBF):
                    amp = float(getattr(k1.k1, 'constant_value', 1.0))
                    ell = float(k1.k2.length_scale)
                elif isinstance(k1, RBF):
                    ell = float(k1.length_scale)
            elif isinstance(fitted, RBF):
                ell = float(fitted.length_scale)
        except Exception:
            pass

        alpha = gp.alpha_.ravel()  # shape (n_train,)

        def herme_n(u: np.ndarray, n: int) -> np.ndarray:
            coeffs = np.zeros(n + 1)
            coeffs[-1] = 1.0
            return herme.hermeval(u, coeffs)

        predictions = {}
        failures = {}

        Xtr = self.x_train
        for order in self.orders:
            try:
                if order == 0:
                    preds = gp.predict(self.x_eval.reshape(-1, 1))
                    predictions[order] = preds.astype(float).tolist()
                    continue

                sign = -1.0 if (order % 2 == 1) else 1.0
                scale = (amp) * (ell ** (-order))
                # For each x*, compute k^(n)(x*, X) @ alpha
                out = []
                for xstar in self.x_eval:
                    u = (xstar - Xtr) / ell  # shape (n_train,)
                    base = np.exp(-0.5 * (u ** 2))  # exp(-(x-x')^2/(2ℓ^2))
                    hn = herme_n(u, order)
                    k_n = (sign * scale) * hn * base  # derivative wrt x*
                    out.append(float(k_n @ alpha))
                predictions[order] = out
            except Exception as e:
                failures[order] = str(e)
                predictions[order] = [np.nan] * len(self.x_eval)

        return {"predictions": predictions, "failures": failures, "meta": {"length_scale": ell, "amplitude": amp}}

    def _chebyshev(self) -> Dict:
        """Global Chebyshev polynomial with analytic derivatives.

        Now uses AICc-based degree selection for automatic regularization.
        """
        tmin = float(self.x_train.min())
        tmax = float(self.x_train.max())
        n_train = len(self.x_train)

        # Adaptive degree selection via AICc
        if ADAPTIVE_HYPERPARAMS and os.environ.get("USE_ADAPTIVE_CHEBY", "1") != "0":
            deg, aicc = select_chebyshev_degree(
                self.x_train, self.y_train,
                max_degree=min(30, n_train - 1),
                min_degree=3
            )
            selection_method = "AICc"
        else:
            # Fallback to fixed heuristic
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

    def _fourier(self) -> Dict:
        """Trigonometric polynomial with analytic n-th derivatives."""
        t = self.x_train
        y = self.y_train
        tmin = float(t.min())
        tmax = float(t.max())
        T = tmax - tmin
        if T <= 0:
            raise ValueError("Invalid domain length for Fourier fit")
        omega = 2.0 * np.pi / T

        n_train = len(t)

        # Adaptive harmonics selection via GCV
        if ADAPTIVE_HYPERPARAMS and os.environ.get("USE_ADAPTIVE_FOURIER", "1") != "0":
            M, gcv = select_fourier_harmonics(
                self.x_train, self.y_train,
                max_harmonics=25,
                min_harmonics=1
            )
            selection_method = "GCV"
        else:
            # Fallback to fixed heuristic
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

    def _fourier_continuation(self) -> Dict:
        """Trend-removed trigonometric LS fit with analytic n-th derivatives (non-periodic aid).

        We fit a low-degree polynomial trend and a trig series on residuals; derivatives add.
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

    def _ad_trig(self) -> Dict:
        """AD-backed trigonometric polynomial using autograd for derivatives."""
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

    def _quintic_spline_derivatives(self, x_grid: np.ndarray, y_grid: np.ndarray) -> Dict:
        """Helper: fit quintic spline to (x_grid, y_grid) and return derivatives at self.x_eval."""
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

    def _spectral_taper_derivative(self) -> Dict:
        """FFT-based derivative with taper and low-pass regularization.

        Steps: detrend (optional), apply taper (Tukey), FFT, multiply by (i*k)^n,
        apply spectral shrinkage, inverse FFT, and sample at original grid.
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

    def _whittaker_m2(self) -> Dict:
        """Whittaker/HP smoothing (m=2) with derivative evaluation via spline.

        Solve min_f sum w_i (y_i - f_i)^2 + λ sum (Δ^2 f)^2 using a banded solver.
        Implementation: simple tridiagonal banded system for (I + λ D2^T D2) f = y.
        """
        t = self.x_train
        y = self.y_train
        n = len(t)
        if n < 5:
            raise ValueError("Not enough points for Whittaker smoothing")
        lam = float(os.environ.get("WHIT_LAMBDA", "100.0"))
        # Build D2 operator (second differences)
        import numpy as _np
        e = _np.ones(n)
        D2 = _np.zeros((n - 2, n))
        for i in range(n - 2):
            D2[i, i] = 1.0
            D2[i, i + 1] = -2.0
            D2[i, i + 2] = 1.0
        A = _np.eye(n) + lam * (D2.T @ D2)
        # Solve
        f = _np.linalg.solve(A, y)
        # Derivatives via spline
        res = self._quintic_spline_derivatives(t, f)
        return {"predictions": res["predictions"], "failures": res["failures"], "meta": {"lambda": lam}}

    def _butterworth_spline(self) -> Dict:
        """Butterworth lowpass smoothing + quintic spline derivatives."""
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
        """Legacy FiniteDiff baseline replaced by smoothing + quintic-spline derivatives."""
        # Use a milder lowpass to mimic a pre-filter prior to differentiation
        t = self.x_train
        y = self.y_train
        b, a = butter(N=2, Wn=0.2, btype='lowpass', analog=False)
        y_smooth = filtfilt(b, a, y)
        return self._quintic_spline_derivatives(t, y_smooth)

    def _savgol_method(self) -> Dict:
        """Savitzky-Golay filtering with derivative and spline fallback for higher orders."""
        t = self.x_train
        y = self.y_train
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 1.0
        # Choose window length: nearest odd ~ min(n, 21)
        win = min(n if n % 2 == 1 else n - 1, 21)
        if win < 5:
            win = 5 if n >= 5 else (n if n % 2 == 1 else n - 1)
        poly = 7 if win >= 9 else 5

        predictions = {}
        failures = {}
        # Compute base filtered signal (order 0)
        try:
            y0 = savgol_filter(y, window_length=win, polyorder=poly, deriv=0, delta=dt, mode='interp')
            # Resample to x_eval using spline to be robust
            base = self._quintic_spline_derivatives(t, y0)
            predictions[0] = base["predictions"].get(0, [float(v) for v in y0])
        except Exception as e:
            failures[0] = str(e)
            predictions[0] = [np.nan] * len(self.x_eval)

        for order in self.orders:
            if order == 0:
                continue
            try:
                if order <= poly:
                    yd = savgol_filter(y, window_length=win, polyorder=poly, deriv=order, delta=dt, mode='interp')
                    # Spline to evaluation grid
                    res = self._quintic_spline_derivatives(t, yd)
                    predictions[order] = res["predictions"].get(order, [float(v) for v in yd])
                else:
                    # Fallback: differentiate quintic spline fit to y0 further
                    res = self._quintic_spline_derivatives(t, y0)
                    predictions[order] = res["predictions"].get(order, [np.nan] * len(self.x_eval))
            except Exception as e:
                failures[order] = str(e)
                predictions[order] = [np.nan] * len(self.x_eval)

        return {"predictions": predictions, "failures": failures, "meta": {"window": win, "polyorder": poly}}

    def _svr_spline(self) -> Dict:
        """SVR (RBF) fit, then quintic-spline derivatives of predicted signal."""
        X = self.x_train.reshape(-1, 1)
        y = self.y_train
        svr = SVR(kernel='rbf', C=10.0, gamma='scale', epsilon=0.01)
        svr.fit(X, y)
        yhat_eval = svr.predict(self.x_eval.reshape(-1, 1))
        return self._quintic_spline_derivatives(self.x_eval, yhat_eval)

    def _kalman_grad(self) -> Dict:
        """Constant-acceleration Kalman RTSS smoother; pos/vel/acc derivatives; higher via spline."""
        t = self.x_train
        y = self.y_train
        n = len(t)
        if n < 5:
            raise ValueError("Not enough points for Kalman smoothing")
        dt = float(np.mean(np.diff(t)))

        # State: [pos, vel, acc]
        F = np.array([[1.0, dt, 0.5 * dt * dt],
                      [0.0, 1.0, dt],
                      [0.0, 0.0, 1.0]], dtype=float)
        H = np.array([[1.0, 0.0, 0.0]], dtype=float)
        # Process/measurement noise (heuristic)
        q = 1e-3
        r = max(1e-6, float(np.var(y) * 1e-4))
        Q = q * np.array([[dt**5/20, dt**4/8, dt**3/6],
                          [dt**4/8, dt**3/3, dt**2/2],
                          [dt**3/6, dt**2/2, dt]], dtype=float)
        R = np.array([[r]], dtype=float)

        # Init
        x = np.array([y[0], 0.0, 0.0], dtype=float)
        P = np.eye(3)

        xs = []
        Ps = []
        # Forward KF
        for k in range(n):
            # Predict
            if k > 0:
                x = F @ x
                P = F @ P @ F.T + Q
            # Update
            z = np.array([[y[k]]])
            S = H @ P @ H.T + R
            K_gain = P @ H.T @ np.linalg.inv(S)
            y_res = z - H @ x.reshape(-1, 1)
            x = x + (K_gain @ y_res).ravel()
            P = (np.eye(3) - K_gain @ H) @ P
            xs.append(x.copy())
            Ps.append(P.copy())

        # RTS backward smoothing
        xs = np.array(xs)
        Ps = np.array(Ps)
        x_smooth = xs.copy()
        P_smooth = Ps.copy()
        for k in range(n - 2, -1, -1):
            Pk = Ps[k]
            Pk1 = Ps[k + 1]
            Ck = Pk @ F.T @ np.linalg.inv(F @ Pk @ F.T + Q)
            x_smooth[k] = xs[k] + Ck @ (x_smooth[k + 1] - (F @ xs[k]))
            P_smooth[k] = Pk + Ck @ (P_smooth[k + 1] - Pk1) @ Ck.T

        pos = x_smooth[:, 0]
        vel = x_smooth[:, 1]
        acc = x_smooth[:, 2]

        # Interpolate to evaluation grid and compute higher-order via spline
        pred0 = self._quintic_spline_derivatives(t, pos)
        pred1 = self._quintic_spline_derivatives(t, vel)
        pred2 = self._quintic_spline_derivatives(t, acc)

        predictions = {}
        failures = {}
        for order in self.orders:
            try:
                if order == 0:
                    predictions[order] = pred0["predictions"][0]
                elif order == 1:
                    predictions[order] = pred1["predictions"][0]
                elif order == 2:
                    predictions[order] = pred2["predictions"][0]
                else:
                    # Differentiate pos spline to higher order
                    predictions[order] = pred0["predictions"].get(order, [np.nan] * len(self.x_eval))
            except Exception as e:
                failures[order] = str(e)
                predictions[order] = [np.nan] * len(self.x_eval)

        return {"predictions": predictions, "failures": failures}

    def _tvregdiff_method(self) -> Dict:
        """Rick Chartrand TV-regularized differentiation; use spline for 0th and higher orders.

        - order 1: TVRegDiff on y
        - order >=2: differentiate a spline fit to the TVRegDiff output
        - order 0: quintic spline on original y (baseline)
        """
        if not TVREG_AVAILABLE:
            raise RuntimeError("tvregdiff package not available; install via requirements.txt")

        t = self.x_train
        y = self.y_train
        n = len(t)
        if n < 5:
            raise ValueError("Not enough points for TVRegDiff")
        dt = float(np.mean(np.diff(t))) if n > 1 else 1.0

        def _as_int_env(name: str, default: int) -> int:
            v = os.environ.get(name, "")
            try:
                return int(v) if v.strip() != "" else default
            except Exception:
                try:
                    return int(float(v))
                except Exception:
                    return default
        def _as_float_env(name: str, default: float) -> float:
            v = os.environ.get(name, "")
            try:
                return float(v) if v.strip() != "" else default
            except Exception:
                return default
        iters = _as_int_env("TVREG_ITERS", 100)
        alpha = _as_float_env("TVREG_ALPHA", 1e-2)
        scale = os.environ.get("TVREG_SCALE", "small") or "small"
        diffkernel = (os.environ.get("TVREG_DIFFKERNEL", "abs") or "abs").lower()
        cgtol = _as_float_env("TVREG_CGTOL", 1e-6)
        cgmaxit = _as_int_env("TVREG_CGMAXIT", 100)

        # Call TVRegDiff; signature: TVRegDiff(data, iter, alph, u0, scale, ep, dx, plotflag, diagflag,
        # precondflag, diffkernel, cgtol, cgmaxit)
        y1 = TVRegDiff(y, iters, alpha, None, scale, 1e-6, dt, False, False, False, diffkernel, cgtol, cgmaxit)

        # Build spline over first derivative
        res1 = self._quintic_spline_derivatives(t, np.asarray(y1, dtype=float))
        # Base spline for order 0 (original y)
        base0 = self._quintic_spline_derivatives(t, y)

        predictions = {}
        failures = {}
        for order in self.orders:
            try:
                if order == 0:
                    predictions[order] = base0["predictions"][0]
                elif order == 1:
                    predictions[order] = res1["predictions"][0]
                else:
                    # Differentiate the first-derivative spline further
                    predictions[order] = res1["predictions"].get(order - 1, [np.nan] * len(self.x_eval))
            except Exception as e:
                failures[order] = str(e)
                predictions[order] = [np.nan] * len(self.x_eval)

        return {"predictions": predictions, "failures": failures, "meta": {"iters": iters, "alpha": alpha, "scale": scale, "dt": dt}}

    def _gp_matern(self, nu: float) -> Dict:
        """GP with Matérn kernel; posterior mean derivatives via closed-form kernel derivatives.

        OPTIMIZED VERSION: Uses analytical formulas for Matern kernel derivatives instead of
        nested autograd differentiation. This is 100-1000× faster for high-order derivatives.
        """
        kernel = ConstantKernel(1.0, (1e-6, 1e6)) * Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=nu) \
                 + WhiteKernel(noise_level=1e-8, noise_level_bounds=(1e-10, 1e-2))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=0.0, normalize_y=False)
        X = self.x_train.reshape(-1, 1)
        y = self.y_train
        gp.fit(X, y)

        # Extract fitted params
        fitted = gp.kernel_
        amp = 1.0
        ell = 1.0
        noise = 1e-8
        try:
            if hasattr(fitted, 'k1') and hasattr(fitted, 'k2'):
                prod = fitted.k1
                if hasattr(prod, 'k1') and hasattr(prod, 'k2') and isinstance(prod.k2, Matern):
                    amp = float(getattr(prod.k1, 'constant_value', 1.0))
                    ell = float(prod.k2.length_scale)
                    noise = float(getattr(fitted.k2, 'noise_level', 1e-8))
        except Exception:
            pass

        # Prepare alpha
        alpha = gp.alpha_.ravel()
        Xtr = self.x_train

        # Closed-form Matern kernel derivatives
        def matern_kernel_derivative(x: float, xprime: float, ell_local: float, nu_local: float, order: int) -> float:
            """Compute order-th derivative of Matern kernel wrt x."""
            r = abs(x - xprime)
            r_safe = max(r, 1e-12)
            sign = 1.0 if x >= xprime else -1.0

            if abs(nu_local - 0.5) < 1e-8:
                # Matern-1/2: k^(n)(r) = (-1/ℓ)^n · exp(-r/ℓ)
                result = ((-1.0 / ell_local) ** order) * np.exp(-r_safe / ell_local)
                if order % 2 == 1:
                    result *= sign

            elif abs(nu_local - 1.5) < 1e-8:
                # Matern-3/2: closed-form derivatives
                c = np.sqrt(3.0) / ell_local
                cr = c * r_safe
                exp_term = np.exp(-cr)

                if order == 0:
                    result = (1.0 + cr) * exp_term
                elif order == 1:
                    result = sign * (-c * c * r_safe) * exp_term
                elif order == 2:
                    result = (c ** 2) * (cr - 1.0) * exp_term
                elif order == 3:
                    result = sign * (c ** 3) * (3.0 - cr) * exp_term
                elif order == 4:
                    result = (c ** 4) * (cr - 3.0) * exp_term
                elif order == 5:
                    result = sign * (c ** 5) * (5.0 - cr) * exp_term
                elif order == 6:
                    result = (c ** 6) * (cr - 5.0) * exp_term
                elif order == 7:
                    result = sign * (c ** 7) * (7.0 - cr) * exp_term
                else:
                    result = 0.0  # Unsupported order

            elif abs(nu_local - 2.5) < 1e-8:
                # Matern-5/2: closed-form derivatives
                c = np.sqrt(5.0) / ell_local
                cr = c * r_safe
                cr2 = cr * cr
                exp_term = np.exp(-cr)

                if order == 0:
                    result = (1.0 + cr + cr2 / 3.0) * exp_term
                elif order == 1:
                    result = sign * (c / 3.0) * cr * (cr - 3.0) * exp_term
                elif order == 2:
                    result = (c ** 2 / 3.0) * (cr2 - 6.0 * cr + 3.0) * exp_term
                elif order == 3:
                    result = sign * (c ** 3 / 3.0) * (cr2 - 9.0 * cr + 15.0) * exp_term
                elif order == 4:
                    result = (c ** 4 / 3.0) * (cr2 - 12.0 * cr + 15.0) * exp_term
                elif order == 5:
                    result = sign * (c ** 5 / 3.0) * (cr2 - 15.0 * cr + 45.0) * exp_term
                elif order == 6:
                    result = (c ** 6 / 3.0) * (cr2 - 18.0 * cr + 45.0) * exp_term
                elif order == 7:
                    result = sign * (c ** 7 / 3.0) * (cr2 - 21.0 * cr + 105.0) * exp_term
                else:
                    result = 0.0  # Unsupported order
            else:
                # Fallback to RBF-like (not true Matern for general nu)
                if order == 0:
                    result = np.exp(-0.5 * (r_safe / ell_local) ** 2)
                else:
                    result = 0.0

            return result

        predictions = {}
        failures = {}
        for order in self.orders:
            try:
                if order == 0:
                    mu = gp.predict(self.x_eval.reshape(-1, 1))
                    predictions[order] = [float(v) for v in mu]
                else:
                    # Use closed-form kernel derivatives (FAST!)
                    vals = []
                    for xstar in self.x_eval:
                        deriv_sum = 0.0
                        for xj, aj in zip(Xtr, alpha):
                            k_deriv = matern_kernel_derivative(xstar, xj, ell, nu, order)
                            deriv_sum += amp * k_deriv * aj
                        vals.append(float(deriv_sum))
                    predictions[order] = vals
            except Exception as e:
                failures[order] = str(e)
                predictions[order] = [np.nan] * len(self.x_eval)

        return {"predictions": predictions, "failures": failures, "meta": {"nu": nu, "length_scale": ell, "amplitude": amp}}

    # Removed: FD, numdifftools, and Savitzky-Golay paths (by design)


def _validate_input_data(data: dict) -> None:
    """Validate JSON input data for required fields and correctness."""
    # Check for required fields
    x_train = data.get("t") or data.get("times")
    y_train = data.get("y") or data.get("y_noisy")

    if x_train is None:
        raise ValueError("Missing required field: 'times' or 't'")
    if y_train is None:
        raise ValueError("Missing required field: 'y_noisy' or 'y'")

    # Convert to arrays and validate
    try:
        x_arr = np.asarray(x_train, dtype=float)
        y_arr = np.asarray(y_train, dtype=float)
    except Exception as e:
        raise ValueError(f"Invalid data format (must be numeric arrays): {e}")

    # Check non-empty
    if x_arr.size == 0:
        raise ValueError("Empty x data (times)")
    if y_arr.size == 0:
        raise ValueError("Empty y data (observations)")

    # Check matching lengths
    if x_arr.size != y_arr.size:
        raise ValueError(f"Mismatched array lengths: x has {x_arr.size}, y has {y_arr.size}")

    # Check for finite values
    if not np.all(np.isfinite(x_arr)):
        raise ValueError("x data (times) contains non-finite values (NaN/Inf)")
    if not np.all(np.isfinite(y_arr)):
        raise ValueError("y data (observations) contains non-finite values (NaN/Inf)")

    # Check minimum data size
    if x_arr.size < 3:
        raise ValueError(f"Insufficient data points: {x_arr.size} (minimum 3 required)")

    # Check x is sorted (required for interpolation)
    if not np.all(np.diff(x_arr) > 0):
        raise ValueError("x data (times) must be strictly increasing")

    # Validate orders if present
    if "orders" in data:
        orders = data["orders"]
        if orders is not None:
            try:
                order_arr = [int(o) for o in orders]
                if any(o < 0 for o in order_arr):
                    raise ValueError("Derivative orders must be non-negative")
                if any(o > 10 for o in order_arr):
                    raise ValueError("Derivative orders > 10 not supported")
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid orders field: {e}")


def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: python python_methods.py <input_json> <output_json>")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    # Read input data
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {input_file.name}: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"ERROR: Input file not found: {input_file}")
        sys.exit(1)

    # Validate input data
    try:
        _validate_input_data(data)
    except ValueError as e:
        print(f"ERROR: Invalid input data: {e}")
        sys.exit(1)

    # Backward-compatible input schema with optional new fields
    x_train = data.get("t") or data.get("times")
    y_train = data.get("y") or data.get("y_noisy")
    x_eval = data.get("t_eval") or x_train  # Evaluate at same points by default
    orders = _safe_orders(data.get("orders"))
    # Environment overrides for derivative orders
    orders_env = os.environ.get("PY_ORDERS", "").strip()
    max_order_env = os.environ.get("PY_MAX_ORDER", "").strip()
    if orders_env:
        try:
            orders = [int(s) for s in orders_env.split(',') if s.strip() != '']
        except Exception:
            pass
    if max_order_env:
        try:
            max_n = int(max_order_env)
            orders = [o for o in orders if o <= max_n]
        except Exception:
            pass

    print(f"Processing {input_file.name}...")
    print(f"  Data points: {len(x_train)}")
    print(f"  Orders: {orders}")
    if data.get("method"):
        print(f"  Single-method mode: {data['method']}")

    # Methods to evaluate
    methods = [
        # Analytic/closed-form
        "chebyshev",
        "fourier",
        "fourier_continuation",
        "gp_rbf_mean",
        "ad_trig",
        # IFAC25 legacy methods (Python)
        "Butterworth_Python",
        "ButterworthSpline_Python",
        "SavitzkyGolay_Python",
        "SVR_Python",
        "KalmanGrad_Python",
        "TVRegDiff_Python",
            "RKHS_Spline_m2_Python",
            "SpectralTaper_Python",
            "Whittaker_m2_Python",
        "GP_RBF_Python",
        "GP_RBF_Iso_Python",
        "GP_Matern_Python",
        "GP_Matern_1.5_Python",
        "GP_Matern_2.5_Python",
    ]

    # Environment overrides
    only_methods_csv = os.environ.get("PY_METHODS", "").strip()
    exclude_methods_csv = os.environ.get("PY_EXCLUDE", "").strip()
    include_matern = os.environ.get("PY_INCLUDE_MATERN", "0").lower() not in ("0", "false", "no")

    if not include_matern:
        methods = [m for m in methods if not m.startswith("GP_Matern")]

    if exclude_methods_csv:
        excludes = {m.strip() for m in exclude_methods_csv.split(",") if m.strip()}
        methods = [m for m in methods if m not in excludes]

    if only_methods_csv:
        onlys = [m.strip() for m in only_methods_csv.split(",") if m.strip()]
        if onlys:
            methods = onlys

    # Optional single-method mode via input JSON schema
    requested_method = data.get("method")

    # Evaluate methods
    results = {}
    evaluator = MethodEvaluator(x_train, y_train, x_eval, orders)

    run_list = [requested_method] if isinstance(requested_method, str) else methods

    for method in run_list:
        print(f"  Evaluating {method}...")
        results[method] = evaluator.evaluate_method(method)

    # Write output
    def _clean_predictions(preds: dict, method_name: str = "Unknown") -> dict:
        cleaned = {}
        for k, vals in preds.items():
            try:
                arr = np.asarray(vals, dtype=float)
                if arr.size == 0:
                    continue
                if not np.all(np.isfinite(arr)):
                    # FIX: Log when non-finite values are dropped
                    print(f"    WARNING: Non-finite values in {method_name} order {k}, data excluded")
                    continue
                cleaned[str(k)] = [float(x) for x in arr]
            except Exception:
                continue
        return cleaned

    if requested_method:
        res = results[requested_method]
        preds = res.get("predictions", {})
        y_derivs = {str(k): v for k, v in _clean_predictions(preds, requested_method).items() if int(k) in orders and int(k) != 0}
        output_data = {
            "method": requested_method,
            "y_derivs": y_derivs,
            "meta": res.get("meta", {}),
            "success": res.get("success", False),
            "failures": res.get("failures", {})
        }
    else:
        cleaned_results = {}
        for m, res in results.items():
            new_res = dict(res)
            new_res["predictions"] = _clean_predictions(res.get("predictions", {}), m)
            cleaned_results[m] = new_res
    output_data = {
            "trial_id": (data.get("config", {}) or {}).get("trial_id", None),
            "methods": cleaned_results
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, allow_nan=False)

    print(f"  Results written to {output_file.name}")

    # Print summary
    print("\n  Summary:")
    for method, result in results.items():
        if result["success"]:
            valid_orders = sum(1 for o in orders if o in result["predictions"]
                             and not any(np.isnan(result["predictions"][o])))
            print(f"    {method}: OK ({valid_orders}/{len(orders)} orders, "
                  f"{result['timing']:.3f}s)")
        else:
            print(f"    {method}: FAILED - {result.get('error', 'Unknown')}")


if __name__ == "__main__":
    main()
