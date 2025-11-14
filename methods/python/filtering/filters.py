"""
Filtering-Based Methods for Derivative Estimation

This module implements filtering and smoothing-based derivative estimation methods:
1. Whittaker smoothing (m=2 penalty) with spline derivatives
2. Savitzky-Golay filtering with polynomial differentiation
3. Kalman RTS smoother (constant-acceleration model)
4. Total Variation Regularized Differentiation (TVRegDiff)
"""

from typing import Dict
import numpy as np
import os

from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline

# Import base class from common utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import MethodEvaluator, TVREG_AVAILABLE

if TVREG_AVAILABLE:
    from tvregdiff import TVRegDiff


class FilteringMethods(MethodEvaluator):
    """
    Filtering and smoothing methods for derivative estimation.

    Inherits from base MethodEvaluator and implements filtering-specific methods.
    """

    def evaluate_method(self, method_name: str) -> Dict:
        """
        Dispatch to the appropriate filtering method based on method name.

        Args:
            method_name: Name of method to evaluate. Supported:
                - "Whittaker_m2_Python": Whittaker/HP smoothing (m=2)
                - "SavitzkyGolay_Python": Savitzky-Golay filtering (fixed window=21)
                - "SavitzkyGolay_Adaptive_Python": Adaptive S-G with noise-dependent window
                - "Kalman-Gradient": Kalman RTS smoother
                - "TVRegDiff-Python": TV-regularized differentiation

        Returns:
            Dictionary with predictions, failures, and metadata
        """
        if method_name == "Whittaker_m2_Python":
            return self._whittaker_m2()
        elif method_name == "SavitzkyGolay_Python":
            return self._savgol_method()
        elif method_name == "SavitzkyGolay_Adaptive_Python":
            return self._savgol_adaptive_method()
        elif method_name == "Kalman-Gradient":
            return self._kalman_grad()
        elif method_name == "TVRegDiff-Python":
            return self._tvregdiff_method()
        else:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": f"Unknown filtering method: {method_name}"}
            }

    def _quintic_spline_derivatives(self, x_grid: np.ndarray, y_grid: np.ndarray) -> Dict:
        """Helper: fit quintic spline to (x_grid, y_grid) and return derivatives at self.x_eval.

        This is an internal helper method used by filtering methods to compute derivatives
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

    def _whittaker_m2(self) -> Dict:
        """Whittaker/HP smoothing (m=2) with derivative evaluation via spline.

        Solve min_f sum w_i (y_i - f_i)^2 + λ sum (Δ^2 f)^2 using a banded solver.
        Implementation: simple tridiagonal banded system for (I + λ D2^T D2) f = y.

        The smoothing parameter λ can be controlled via environment variable:
        - WHIT_LAMBDA: Smoothing parameter (default 100.0)

        Returns:
            Dictionary containing:
                predictions: Dict mapping derivative order to list of predictions
                failures: Dict of any errors encountered
                meta: Smoothing parameter lambda
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

    def _savgol_method(self) -> Dict:
        """Savitzky-Golay filtering with derivative and spline fallback for higher orders.

        Uses scipy's savgol_filter for computing derivatives up to the polynomial order,
        with quintic spline interpolation to the evaluation grid.

        Returns:
            Dictionary containing:
                predictions: Dict mapping derivative order to list of predictions
                failures: Dict of any errors encountered
                meta: Window length and polynomial order used
        """
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

    def _savgol_adaptive_method(self) -> Dict:
        """Adaptive Savitzky-Golay filtering with noise-dependent window sizing.

        This method addresses the bias floors observed in fixed-window S-G by adapting
        the window size based on estimated noise level and signal roughness. Uses larger
        windows for higher derivative orders to combat variance growth.

        Key features:
        - Noise-adaptive window sizing using MAD estimator
        - Polynomial order tiers: r≤3→p=7; 4-5→p=9; 6-7→p=11
        - Per-order optimization (w_r, p_r) for each derivative
        - Window scaling: h* ∝ (σ²/ρ²)^{1/(2p+3)} from MISE minimization

        Note: Derivative estimates are optimized independently for each order and are not
        guaranteed to be mathematically consistent (i.e., d/dt[f^(r)] ≠ f^(r+1)).

        Returns:
            Dictionary containing:
                predictions: Dict mapping derivative order to list of predictions
                failures: Dict of any errors encountered
                meta: Dict of window/polyorder used for each derivative order
        """
        t = self.x_train
        y = self.y_train
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 1.0

        # Robust noise estimation using MAD of first differences
        # Variance of y[i] - y[i-1] is 2σ², so normalize by sqrt(2)
        dy = np.diff(y)
        sigma_hat = np.median(np.abs(dy - np.median(dy))) / (0.6745 * np.sqrt(2.0))

        # Roughness estimation using 4th-order differences (proxy for signal smoothness)
        # Normalized by dt^4 for scale consistency
        if n >= 5:
            d4 = np.diff(y, n=4)
            rho_hat = np.sqrt(np.mean(d4**2)) / ((dt**4) + 1e-24)
        else:
            # Fallback for very small signals
            rho_hat = 1.0

        # Calibration constants c_{p,r} - empirically tuned for N≈100
        # These balance bias-variance tradeoff for different (polyorder, deriv_order) combinations
        # Start with conservative values; can be refined through grid search
        c_pr = {
            (7, 0): 1.0, (7, 1): 1.1, (7, 2): 1.2, (7, 3): 1.3,
            (9, 0): 1.0, (9, 1): 1.1, (9, 2): 1.2, (9, 3): 1.3, (9, 4): 1.4, (9, 5): 1.5,
            (11, 0): 1.0, (11, 1): 1.1, (11, 2): 1.2, (11, 3): 1.3, (11, 4): 1.4,
            (11, 5): 1.5, (11, 6): 1.6, (11, 7): 1.7,
        }

        # Window cap for safety: N/3 for N=101 gives max window of 33
        max_window = max(5, int(n / 3))
        if max_window % 2 == 0:
            max_window -= 1

        predictions = {}
        failures = {}
        meta = {}

        for order in self.orders:
            try:
                # Determine polynomial order based on derivative order (tiered approach)
                if order <= 3:
                    p = 7
                elif order <= 5:
                    p = 9
                else:
                    p = 11

                # Get calibration constant
                c = c_pr.get((p, order), 1.0 + 0.1 * order)

                # Compute optimal window via plug-in rule
                # h* ∝ (σ²/ρ²)^{1/(2p+3)}
                ratio = max(sigma_hat, 1e-24)**2 / max(rho_hat, 1e-24)**2
                h_star = c * (ratio ** (1.0 / (2*p + 3)))

                # Convert to window length in samples
                w_ideal = int(2 * np.floor(h_star / max(dt, 1e-24)) + 1)

                # Apply constraints
                w = max(w_ideal, p + 3)  # Need at least p+3 for numerical headroom
                w = min(w, max_window)   # Safety cap
                w = min(w, n if n % 2 == 1 else n - 1)  # Can't exceed data size

                # Ensure odd
                if w % 2 == 0:
                    w -= 1

                # Final check: window must be > polyorder
                if w <= p:
                    w = p + 2 if p + 2 <= n else (p + 1 if p + 1 <= n else p)
                    if w % 2 == 0:
                        w += 1
                    if w > n:
                        w = n if n % 2 == 1 else n - 1

                # Apply Savitzky-Golay filter for this derivative order
                yd = savgol_filter(y, window_length=w, polyorder=p, deriv=order, delta=dt, mode='interp')

                # Resample to evaluation grid using spline
                res = self._quintic_spline_derivatives(t, yd)
                predictions[order] = res["predictions"].get(order, [float(v) for v in yd])

                # Store parameters used
                meta[order] = {"window": w, "polyorder": p, "noise_est": sigma_hat, "roughness_est": rho_hat}

            except Exception as e:
                failures[order] = str(e)
                predictions[order] = [np.nan] * len(self.x_eval)
                meta[order] = {"error": str(e)}

        return {"predictions": predictions, "failures": failures, "meta": meta}

    def _kalman_grad(self) -> Dict:
        """Constant-acceleration Kalman RTSS smoother; pos/vel/acc derivatives; higher via spline.

        Uses a Kalman filter with Rauch-Tung-Striebel smoother (RTS) for backward pass.
        State vector: [position, velocity, acceleration]
        Extracts 0th, 1st, 2nd derivatives from the state, higher orders via spline.

        Returns:
            Dictionary containing:
                predictions: Dict mapping derivative order to list of predictions
                failures: Dict of any errors encountered
        """
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

        Environment variables for tuning:
        - TVREG_ITERS: Number of iterations (default 100)
        - TVREG_ALPHA: Regularization parameter (default 1e-2)
        - TVREG_SCALE: Scaling mode 'small' or 'large' (default 'small')
        - TVREG_DIFFKERNEL: Differentiation kernel 'abs' or other (default 'abs')
        - TVREG_CGTOL: CG solver tolerance (default 1e-6)
        - TVREG_CGMAXIT: CG max iterations (default 100)

        Returns:
            Dictionary containing:
                predictions: Dict mapping derivative order to list of predictions
                failures: Dict of any errors encountered
                meta: TVRegDiff parameters used
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
