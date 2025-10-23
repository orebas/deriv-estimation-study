"""
Adaptive AAA Methods for Derivative Estimation

This module implements AAA (Adaptive Antoulas-Anderson) rational approximation methods
with adaptive tolerance selection for derivative estimation:

1. AAA-Python with wavelet-based noise estimation (baryrat implementation)
2. AAA-Python with diff2-based noise estimation (baryrat implementation)
3. AAA-JAX with wavelet noise estimation and automatic differentiation
4. AAA-JAX with diff2 noise estimation and automatic differentiation

The JAX variants use automatic differentiation to compute derivatives of all orders,
while the Python (baryrat) variants use analytical derivatives up to order 2.
"""

from typing import Dict, Callable
import numpy as np

# Import base class from common utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import MethodEvaluator, ADAPTIVE_HYPERPARAMS, BARYRAT_AVAILABLE, JAX_AAA_AVAILABLE

if ADAPTIVE_HYPERPARAMS:
    from hyperparameters import (
        select_aaa_tolerance,
        estimate_noise_auto,
        estimate_noise_diff2
    )

if BARYRAT_AVAILABLE:
    from baryrat import aaa

if JAX_AAA_AVAILABLE:
    import jax
    import jax.numpy as jnp
    from baryrat_jax import aaa as aaa_jax


class AdaptiveMethods(MethodEvaluator):
    """
    Adaptive AAA rational approximation methods for derivative estimation.

    Inherits from base MethodEvaluator and implements AAA-specific methods.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with cache for JAX derivatives."""
        super().__init__(*args, **kwargs)
        self._aaa_jax_all_derivs_cache = None

    def evaluate_method(self, method_name: str) -> Dict:
        """
        Dispatch to the appropriate AAA method based on method name.

        Args:
            method_name: Name of method to evaluate. Supported:
                - "AAA-Python-Adaptive-Wavelet": AAA with wavelet noise estimation
                - "AAA-Python-Adaptive-Diff2": AAA with diff2 noise estimation
                - "AAA-JAX-Adaptive-Wavelet": AAA-JAX with wavelet + AD
                - "AAA-JAX-Adaptive-Diff2": AAA-JAX with diff2 + AD

        Returns:
            Dictionary with predictions, failures, and metadata
        """
        if method_name == "AAA-Python-Adaptive-Wavelet":
            return self._aaa_adaptive_wavelet()
        elif method_name == "AAA-Python-Adaptive-Diff2":
            return self._aaa_adaptive_diff2()
        elif method_name == "AAA-JAX-Adaptive-Wavelet":
            return self._aaa_jax_adaptive_wavelet()
        elif method_name == "AAA-JAX-Adaptive-Diff2":
            return self._aaa_jax_adaptive_diff2()
        else:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": f"Unknown adaptive method: {method_name}"}
            }

    def _aaa_adaptive_base(
        self,
        noise_estimator_func: Callable,
        aaa_func: Callable,
        use_jax_ad: bool,
        implementation_name: str,
        selection_name: str,
        availability_flag: bool,
        availability_error: str
    ) -> Dict:
        """
        Unified helper for all AAA adaptive methods.

        Consolidates common logic across AAA-Adaptive-Wavelet, AAA-Adaptive-Diff2,
        AAA-JAX-Adaptive-Wavelet, and AAA-JAX-Adaptive-Diff2.

        Args:
            noise_estimator_func: Function to estimate noise (e.g., estimate_noise_auto)
            aaa_func: AAA implementation (aaa or aaa_jax)
            use_jax_ad: If True, use JAX AD for all orders; else baryrat eval_deriv (≤ order 2)
            implementation_name: Name for meta (e.g., "baryrat", "JAX-AD")
            selection_name: Noise estimation method for meta (e.g., "wavelet-MAD", "diff2")
            availability_flag: Check if dependencies available (BARYRAT_AVAILABLE or JAX_AAA_AVAILABLE)
            availability_error: Error message if dependencies unavailable

        Returns:
            Standard method result dict with predictions, failures, meta
        """
        if not availability_flag:
            return {
                "predictions": {},
                "failures": {0: availability_error},
                "meta": {"error": availability_error}
            }

        if not ADAPTIVE_HYPERPARAMS:
            return {
                "predictions": {},
                "failures": {0: "hyperparameters module not available"},
                "meta": {"error": "hyperparameters module not found"}
            }

        # Estimate noise and set adaptive tolerance with cap
        tol = select_aaa_tolerance(
            self.y_train,
            noise_estimator_func,
            multiplier=10.0,
            max_tol_fraction=0.1
        )
        σ_hat = noise_estimator_func(self.y_train)  # For logging

        # Fit AAA rational approximation
        try:
            r = aaa_func(self.x_train, self.y_train, tol=tol)
        except Exception as e:
            return {
                "predictions": {},
                "failures": {0: f"AAA fit failed: {str(e)}"},
                "meta": {"tolerance": tol, "noise_estimate": σ_hat}
            }

        predictions = {}
        failures = {}

        for order in self.orders:
            try:
                if order == 0:
                    # Function evaluation
                    vals = r(self.x_eval)
                elif use_jax_ad:
                    # JAX AD: Use Taylor-mode jet for 10× speedup
                    # Import jet-based derivatives (with fallback to nested grad)
                    try:
                        from jax_derivatives_jet import compute_derivatives_at_points
                        USE_JET = True
                    except ImportError:
                        USE_JET = False

                    if USE_JET:
                        # FAST PATH: Compute ALL derivatives 0-max_order at ALL points in one pass
                        # This is ~10× faster than nested grad() calls
                        # Only compute once per AAA fit, then extract the needed order
                        if not hasattr(self, '_aaa_jax_all_derivs_cache') or self._aaa_jax_all_derivs_cache is None:
                            max_order_needed = max(self.orders)
                            # Shape: (max_order+1, n_eval_points)
                            self._aaa_jax_all_derivs_cache = compute_derivatives_at_points(
                                lambda x: r(x),
                                jnp.array(self.x_eval),
                                max_order=max_order_needed
                            )
                        # Extract the specific order we need
                        vals = self._aaa_jax_all_derivs_cache[order, :]
                    else:
                        # SLOW FALLBACK: Nested grad() calls (original implementation)
                        def nth_derivative(f, n):
                            """Compute n-th derivative via nested jax.grad() calls."""
                            result = f
                            for _ in range(n):
                                result = jax.grad(result)
                            return result

                        deriv_func = nth_derivative(lambda x: r(x), order)
                        vals = jnp.array([deriv_func(jnp.array(xi)) for xi in self.x_eval])
                elif order <= 2:
                    # baryrat: only supports derivatives up to order 2
                    vals = r.eval_deriv(self.x_eval, k=order)
                else:
                    # baryrat limitation
                    failures[order] = "baryrat only supports derivatives up to order 2"
                    predictions[order] = [np.nan] * len(self.x_eval)
                    continue

                predictions[order] = [float(v) for v in np.asarray(vals)]
            except Exception as e:
                failures[order] = str(e)
                predictions[order] = [np.nan] * len(self.x_eval)

        meta = {
            "tolerance": float(tol),
            "noise_estimate": float(σ_hat),
            "selection": selection_name,
            "implementation": implementation_name
        }
        if hasattr(r, 'degree'):
            meta["degree"] = r.degree()

        return {"predictions": predictions, "failures": failures, "meta": meta}

    def _aaa_adaptive_wavelet(self) -> Dict:
        """AAA rational approximation with wavelet-based adaptive tolerance.

        Uses the baryrat implementation with wavelet-based (MAD) noise estimation
        for adaptive tolerance selection. Computes derivatives up to order 2 using
        analytical formulas.

        Returns:
            Dictionary containing:
                predictions: Dict mapping derivative order to list of predictions
                failures: Dict of any errors encountered
                meta: Tolerance, noise estimate, and AAA degree
        """
        return self._aaa_adaptive_base(
            noise_estimator_func=estimate_noise_auto,
            aaa_func=aaa,
            use_jax_ad=False,
            implementation_name="baryrat",
            selection_name="wavelet-MAD",
            availability_flag=BARYRAT_AVAILABLE,
            availability_error="baryrat package not available"
        )

    def _aaa_adaptive_diff2(self) -> Dict:
        """AAA rational approximation with 2nd-order difference noise estimation.

        Uses the baryrat implementation with second-order finite difference noise
        estimation for adaptive tolerance selection. Computes derivatives up to
        order 2 using analytical formulas.

        Returns:
            Dictionary containing:
                predictions: Dict mapping derivative order to list of predictions
                failures: Dict of any errors encountered
                meta: Tolerance, noise estimate, and AAA degree
        """
        return self._aaa_adaptive_base(
            noise_estimator_func=estimate_noise_diff2,
            aaa_func=aaa,
            use_jax_ad=False,
            implementation_name="baryrat",
            selection_name="diff2",
            availability_flag=BARYRAT_AVAILABLE,
            availability_error="baryrat package not available"
        )

    def _aaa_jax_adaptive_wavelet(self) -> Dict:
        """AAA-JAX with wavelet noise estimation and automatic differentiation for all orders.

        Uses JAX-based AAA implementation (baryrat_jax) with wavelet-based noise
        estimation. Computes derivatives of all orders using JAX automatic differentiation,
        either via fast Taylor-mode jets or fallback nested grad() calls.

        Returns:
            Dictionary containing:
                predictions: Dict mapping derivative order to list of predictions
                failures: Dict of any errors encountered
                meta: Tolerance, noise estimate, AAA degree, and implementation details
        """
        return self._aaa_adaptive_base(
            noise_estimator_func=estimate_noise_auto,
            aaa_func=aaa_jax,
            use_jax_ad=True,
            implementation_name="JAX-AD",
            selection_name="wavelet-MAD",
            availability_flag=JAX_AAA_AVAILABLE,
            availability_error="JAX or baryrat_jax not available"
        )

    def _aaa_jax_adaptive_diff2(self) -> Dict:
        """AAA-JAX with diff2 noise estimation and automatic differentiation for all orders.

        Uses JAX-based AAA implementation (baryrat_jax) with second-order difference
        noise estimation. Computes derivatives of all orders using JAX automatic
        differentiation, either via fast Taylor-mode jets or fallback nested grad() calls.

        Returns:
            Dictionary containing:
                predictions: Dict mapping derivative order to list of predictions
                failures: Dict of any errors encountered
                meta: Tolerance, noise estimate, AAA degree, and implementation details
        """
        return self._aaa_adaptive_base(
            noise_estimator_func=estimate_noise_diff2,
            aaa_func=aaa_jax,
            use_jax_ad=True,
            implementation_name="JAX-AD",
            selection_name="diff2",
            availability_flag=JAX_AAA_AVAILABLE,
            availability_error="JAX or baryrat_jax not available"
        )
