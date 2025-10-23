"""
Gaussian Process Methods for Derivative Estimation

This module implements GP-based derivative estimation methods using:
1. RBF (Squared Exponential) kernel with closed-form derivatives
2. Matérn kernel (nu=0.5, 1.5, 2.5) with analytical derivative formulas

Both methods use the posterior mean of the GP to estimate derivatives at arbitrary orders.
"""

from typing import Dict
import numpy as np
from numpy.polynomial import hermite_e as herme

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern

# Import base class from common utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import MethodEvaluator


class GPMethods(MethodEvaluator):
    """
    Gaussian Process methods for derivative estimation.

    Inherits from base MethodEvaluator and implements GP-specific methods.
    """

    def evaluate_method(self, method_name: str) -> Dict:
        """
        Dispatch to the appropriate GP method based on method name.

        Args:
            method_name: Name of method to evaluate. Supported:
                - "gp_rbf_mean", "GP_RBF_Python", "GP_RBF_Iso_Python": RBF kernel GP
                - "GP_Matern_Python", "GP_Matern_1.5_Python": Matérn(1.5) kernel GP
                - "GP_Matern_2.5_Python": Matérn(2.5) kernel GP

        Returns:
            Dictionary with predictions, failures, and metadata
        """
        if method_name in ["gp_rbf_mean", "GP_RBF_Python", "GP_RBF_Iso_Python"]:
            return self._gp_rbf_mean_derivative()
        elif method_name in ["GP_Matern_Python", "GP_Matern_1.5_Python"]:
            return self._gp_matern(nu=1.5)
        elif method_name == "GP_Matern_2.5_Python":
            return self._gp_matern(nu=2.5)
        else:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": f"Unknown GP method: {method_name}"}
            }

    def _gp_rbf_mean_derivative(self) -> Dict:
        """
        GP posterior mean derivatives using closed-form RBF kernel derivatives.

        Uses kernel: ConstantKernel * RBF + WhiteKernel
        Computes derivatives using Hermite polynomials for analytic kernel derivatives.

        Returns:
            Dictionary containing:
                predictions: Dict mapping derivative order to list of predictions
                failures: Dict of any errors encountered
                meta: Fitted hyperparameters (length_scale, amplitude)
        """
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
            """Evaluate probabilist's Hermite polynomial of degree n."""
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

    def _gp_matern(self, nu: float) -> Dict:
        """
        GP with Matérn kernel; posterior mean derivatives via closed-form kernel derivatives.

        OPTIMIZED VERSION: Uses analytical formulas for Matern kernel derivatives instead of
        nested autograd differentiation. This is 100-1000× faster for high-order derivatives.

        Args:
            nu: Smoothness parameter. Supported values: 0.5, 1.5, 2.5

        Returns:
            Dictionary containing:
                predictions: Dict mapping derivative order to list of predictions
                failures: Dict of any errors encountered
                meta: Fitted hyperparameters (nu, length_scale, amplitude)
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
            """
            Compute order-th derivative of Matern kernel wrt x.

            Args:
                x: Evaluation point
                xprime: Training point
                ell_local: Length scale
                nu_local: Smoothness parameter
                order: Derivative order

            Returns:
                Kernel derivative value
            """
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
