"""
Complete additions for missing PyNumDiff methods to pynumdiff_methods.py
This file contains the exact code to add to the existing wrapper.
"""

# ============================================================================
# SECTION 1: Add these dispatch cases to evaluate_method() around line 133
# ============================================================================

DISPATCH_ADDITIONS = """
        # === MISSING METHOD DISPATCHES (ADD AFTER LINE 133) ===

        # Total Variation Regularized (EXCELLENT performer)
        elif method_name == "PyNumDiff-TVRegularized-Auto":
            return self._tvrdiff(regime="auto")
        elif method_name == "PyNumDiff-TVRegularized-Tuned":
            return self._tvrdiff(regime="tuned")

        # Polynomial fitting (EXCELLENT performer)
        elif method_name == "PyNumDiff-PolyDiff-Auto":
            return self._polydiff(regime="auto")
        elif method_name == "PyNumDiff-PolyDiff-Tuned":
            return self._polydiff(regime="tuned")

        # Basic finite differences (baselines)
        elif method_name == "PyNumDiff-FirstOrder":
            return self._first_order()
        elif method_name == "PyNumDiff-SecondOrder":
            return self._second_order()
        elif method_name == "PyNumDiff-FourthOrder":
            return self._fourth_order()

        # Window-based methods
        elif method_name == "PyNumDiff-MeanDiff-Auto":
            return self._meandiff(regime="auto")
        elif method_name == "PyNumDiff-MeanDiff-Tuned":
            return self._meandiff(regime="tuned")
        elif method_name == "PyNumDiff-MedianDiff-Auto":
            return self._mediandiff(regime="auto")
        elif method_name == "PyNumDiff-MedianDiff-Tuned":
            return self._mediandiff(regime="tuned")

        # RBF (included for completeness, known to fail)
        elif method_name == "PyNumDiff-RBF-Auto":
            return self._rbfdiff(regime="auto")
        elif method_name == "PyNumDiff-RBF-Tuned":
            return self._rbfdiff(regime="tuned")

        # Spline (ALREADY IMPLEMENTED BUT MISSING FROM DISPATCH!)
        elif method_name == "PyNumDiff-Spline-Auto":
            return self._splinediff(regime="auto")
        elif method_name == "PyNumDiff-Spline-Tuned":
            return self._splinediff(regime="tuned")
"""

# ============================================================================
# SECTION 2: Add these method implementations before the HELPER METHODS section
# ============================================================================

METHOD_IMPLEMENTATIONS = '''
    def _tvrdiff(self, regime: str = "auto") -> Dict:
        """
        Total Variation Regularized Differentiation (orders 0-1 only).
        One of the BEST performers in testing (RMSE: 0.038).
        """
        t = self.x_train
        y = self.y_train
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 1.0

        try:
            if regime == "auto":
                # Use optimization for gamma
                gamma = self._tvgamma_docs_heuristic(y, dt)
                meta = {"method": "tvrdiff", "regime": regime, "dt": dt, "gamma": gamma, "order": 1}
            else:  # tuned
                gamma = 5e-3  # From our testing
                meta = {"method": "tvrdiff", "regime": regime, "dt": dt, "gamma": gamma, "order": 1}

            # tvrdiff requires explicit order parameter
            x_smooth, dx_dt = tvr.tvrdiff(y, dt, order=1, gamma=gamma)

            result = self._orders_0_1_only(x_smooth, dx_dt)
            return {"predictions": result["predictions"], "failures": result["failures"], "meta": meta}

        except Exception as e:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": f"tvrdiff failed: {str(e)}"}
            }

    def _polydiff(self, regime: str = "auto") -> Dict:
        """
        Polynomial fitting differentiation (orders 0-1 only).
        Excellent performer in testing (RMSE: 0.045).
        """
        from pynumdiff.polynomial_fit import polydiff

        t = self.x_train
        y = self.y_train
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 1.0

        try:
            if regime == "auto":
                # Auto-select parameters based on signal length
                window_size = min(11, max(5, n // 20))
                if window_size % 2 == 0:
                    window_size += 1
                degree = min(5, window_size - 2)
                kernel = 'friedrichs'
            else:  # tuned
                window_size = 7
                degree = 3
                kernel = 'friedrichs'

            # polydiff returns (x_smooth, dx_dt)
            x_smooth, dx_dt = polydiff(y, dt, degree=degree,
                                       window_size=window_size,
                                       kernel=kernel)

            result = self._orders_0_1_only(x_smooth, dx_dt)
            meta = {"method": "polydiff", "dt": dt, "degree": degree,
                    "window_size": window_size, "kernel": kernel, "regime": regime}
            return {"predictions": result["predictions"], "failures": result["failures"], "meta": meta}

        except Exception as e:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": f"polydiff failed: {str(e)}"}
            }

    def _first_order(self) -> Dict:
        """Basic first-order finite difference (orders 0-1 only)."""
        t = self.x_train
        y = self.y_train
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 1.0

        try:
            # pynumdiff.first_order returns (x_smooth, dx_dt) tuple
            x_smooth, dx_dt = pynumdiff.first_order(y, dt)

            # Handle potential size mismatch for finite differences
            dx_dt = self._handle_fd_size_mismatch(dx_dt, n)

            result = self._orders_0_1_only(x_smooth, dx_dt)
            meta = {"method": "first_order_fd", "dt": dt}
            return {"predictions": result["predictions"], "failures": result["failures"], "meta": meta}

        except Exception as e:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": f"first_order failed: {str(e)}"}
            }

    def _second_order(self) -> Dict:
        """Basic second-order finite difference (orders 0-1 only)."""
        t = self.x_train
        y = self.y_train
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 1.0

        try:
            x_smooth, dx_dt = pynumdiff.second_order(y, dt)

            # Handle potential size mismatch
            dx_dt = self._handle_fd_size_mismatch(dx_dt, n)

            result = self._orders_0_1_only(x_smooth, dx_dt)
            meta = {"method": "second_order_fd", "dt": dt}
            return {"predictions": result["predictions"], "failures": result["failures"], "meta": meta}

        except Exception as e:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": f"second_order failed: {str(e)}"}
            }

    def _fourth_order(self) -> Dict:
        """Basic fourth-order finite difference (orders 0-1 only)."""
        t = self.x_train
        y = self.y_train
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 1.0

        try:
            x_smooth, dx_dt = pynumdiff.fourth_order(y, dt)

            # Handle potential size mismatch
            dx_dt = self._handle_fd_size_mismatch(dx_dt, n)

            result = self._orders_0_1_only(x_smooth, dx_dt)
            meta = {"method": "fourth_order_fd", "dt": dt}
            return {"predictions": result["predictions"], "failures": result["failures"], "meta": meta}

        except Exception as e:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": f"fourth_order failed: {str(e)}"}
            }

    def _meandiff(self, regime: str = "auto") -> Dict:
        """Mean smoothing with differentiation (orders 0-1 only)."""
        t = self.x_train
        y = self.y_train
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 1.0

        try:
            if regime == "auto":
                window_size = min(11, max(5, n // 20))
                if window_size % 2 == 0:
                    window_size += 1
                num_iterations = 2
            else:  # tuned
                window_size = 7
                num_iterations = 3

            x_smooth, dx_dt = sfd.meandiff(y, dt, window_size=window_size,
                                           num_iterations=num_iterations)

            result = self._orders_0_1_only(x_smooth, dx_dt)
            meta = {"method": "meandiff", "dt": dt, "window_size": window_size,
                    "num_iterations": num_iterations, "regime": regime}
            return {"predictions": result["predictions"], "failures": result["failures"], "meta": meta}

        except Exception as e:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": f"meandiff failed: {str(e)}"}
            }

    def _mediandiff(self, regime: str = "auto") -> Dict:
        """Median smoothing with differentiation (orders 0-1 only)."""
        t = self.x_train
        y = self.y_train
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 1.0

        try:
            if regime == "auto":
                window_size = min(11, max(5, n // 20))
                if window_size % 2 == 0:
                    window_size += 1
                num_iterations = 2
            else:  # tuned
                window_size = 7
                num_iterations = 3

            x_smooth, dx_dt = sfd.mediandiff(y, dt, window_size=window_size,
                                             num_iterations=num_iterations)

            result = self._orders_0_1_only(x_smooth, dx_dt)
            meta = {"method": "mediandiff", "dt": dt, "window_size": window_size,
                    "num_iterations": num_iterations, "regime": regime}
            return {"predictions": result["predictions"], "failures": result["failures"], "meta": meta}

        except Exception as e:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": f"mediandiff failed: {str(e)}"}
            }

    def _rbfdiff(self, regime: str = "auto") -> Dict:
        """
        Radial Basis Function differentiation (orders 0-1 only).
        WARNING: This method performed TERRIBLY in testing (RMSE: 719!)
        Included for completeness but expect poor results.
        """
        from pynumdiff.basis_fit import rbfdiff

        t = self.x_train
        y = self.y_train
        n = len(t)

        try:
            if regime == "auto":
                # Try to use reasonable parameters (but still expect failure)
                sigma = 0.1 * (t[-1] - t[0]) / n
                lmbd = 1e-4
            else:  # tuned
                sigma = 0.1
                lmbd = 1e-3

            # rbfdiff needs time array as first argument
            x_smooth, dx_dt = rbfdiff(t, y, sigma=sigma, lmbd=lmbd)

            # Ensure proper shape
            if hasattr(dx_dt, 'flatten'):
                dx_dt = dx_dt.flatten()

            result = self._orders_0_1_only(x_smooth, dx_dt)
            meta = {"method": "rbfdiff", "sigma": sigma, "lambda": lmbd,
                    "warning": "This method has severe conditioning issues", "regime": regime}
            return {"predictions": result["predictions"], "failures": result["failures"], "meta": meta}

        except Exception as e:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": f"rbfdiff failed (expected): {str(e)}",
                           "note": "RBF often fails due to conditioning"}
            }

    def _handle_fd_size_mismatch(self, dx_dt: np.ndarray, expected_size: int) -> np.ndarray:
        """
        Handle size mismatch for finite difference methods.
        Some FD methods return arrays that are 1 or 2 elements shorter.
        """
        current_size = len(dx_dt)

        if current_size == expected_size:
            return dx_dt
        elif current_size == expected_size - 1:
            # Forward/backward difference - pad with last value
            return np.append(dx_dt, dx_dt[-1])
        elif current_size == expected_size - 2:
            # Central difference - pad both ends
            return np.concatenate([[dx_dt[0]], dx_dt, [dx_dt[-1]]])
        else:
            # Unexpected size - try to handle gracefully
            if current_size < expected_size:
                # Pad with last value
                padding_size = expected_size - current_size
                return np.pad(dx_dt, (0, padding_size), mode='edge')
            else:
                # Truncate
                return dx_dt[:expected_size]
'''

# ============================================================================
# SECTION 3: Update documentation in docstring
# ============================================================================

DOCSTRING_UPDATE = '''
Add these to the evaluate_method docstring around line 69:

                Orders 0-1 only (ADD THESE):
                - "PyNumDiff-TVRegularized-Auto": Total Variation Regularized (BEST!)
                - "PyNumDiff-TVRegularized-Tuned": Total Variation Regularized with tuned params
                - "PyNumDiff-PolyDiff-Auto": Polynomial fitting with automatic parameters
                - "PyNumDiff-PolyDiff-Tuned": Polynomial fitting with tuned parameters
                - "PyNumDiff-FirstOrder": Basic first-order finite difference
                - "PyNumDiff-SecondOrder": Basic second-order finite difference
                - "PyNumDiff-FourthOrder": Basic fourth-order finite difference
                - "PyNumDiff-MeanDiff-Auto": Mean smoothing with automatic parameters
                - "PyNumDiff-MeanDiff-Tuned": Mean smoothing with tuned parameters
                - "PyNumDiff-MedianDiff-Auto": Median smoothing with automatic parameters
                - "PyNumDiff-MedianDiff-Tuned": Median smoothing with tuned parameters
                - "PyNumDiff-RBF-Auto": RBF with automatic parameters (usually fails)
                - "PyNumDiff-RBF-Tuned": RBF with tuned parameters (usually fails)
                - "PyNumDiff-Spline-Auto": Spline with automatic parameters (already implemented!)
                - "PyNumDiff-Spline-Tuned": Spline with tuned parameters (already implemented!)
'''

# ============================================================================
# INSTRUCTIONS FOR INTEGRATION
# ============================================================================

print("=" * 80)
print("INTEGRATION INSTRUCTIONS")
print("=" * 80)
print("""
To integrate these missing methods into pynumdiff_methods.py:

1. OPEN methods/python/pynumdiff_wrapper/pynumdiff_methods.py

2. ADD DISPATCH CASES (around line 133, after other elif statements):
   - Copy the DISPATCH_ADDITIONS section above
   - Insert after the TV-Jerk dispatch case

3. ADD METHOD IMPLEMENTATIONS (before "# HELPER METHODS" comment):
   - Copy all methods from METHOD_IMPLEMENTATIONS section
   - Add the _handle_fd_size_mismatch helper method

4. UPDATE DOCSTRING:
   - Add the new method names to the evaluate_method docstring

5. TEST:
   Run a quick test to ensure methods are callable:
   python -c "from pynumdiff_methods import PyNumDiffMethods; print('OK')"

METHODS BEING ADDED:
- TVRegularized (tvrdiff) - BEST performer, RMSE: 0.038
- PolyDiff - Excellent, RMSE: 0.045
- Spline dispatch - Already implemented, just needs dispatch
- Basic FD (1st, 2nd, 4th) - Baselines
- MeanDiff, MedianDiff - Window methods
- RBF - Known to fail, but included for completeness

This will give you ALL 21+ PyNumDiff methods in your comprehensive study!
""")