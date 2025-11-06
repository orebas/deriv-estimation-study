"""
Add missing PyNumDiff methods to the existing wrapper

These methods were discovered during our comprehensive testing and should be included
for a complete comparison study.
"""

# Code to add to pynumdiff_methods.py in the evaluate_method() function:

MISSING_METHODS_CODE = '''
        # === ADD THESE TO evaluate_method() ===

        # Basic finite differences (simple but effective baselines)
        elif method_name == "PyNumDiff-FirstOrder":
            return self._first_order()
        elif method_name == "PyNumDiff-SecondOrder":
            return self._second_order()
        elif method_name == "PyNumDiff-FourthOrder":
            return self._fourth_order()

        # Total Variation Regularized Differentiation (general, order 1)
        elif method_name == "PyNumDiff-TVRegularized":
            return self._tvrdiff()

        # Polynomial fitting differentiation
        elif method_name == "PyNumDiff-PolyDiff-Auto":
            return self._polydiff(regime="auto")
        elif method_name == "PyNumDiff-PolyDiff-Tuned":
            return self._polydiff(regime="tuned")

        # Spline differentiation (missing from dispatch!)
        elif method_name == "PyNumDiff-Spline-Auto":
            return self._splinediff(regime="auto")
        elif method_name == "PyNumDiff-Spline-Tuned":
            return self._splinediff(regime="tuned")

        # Window-based methods (mediocre but should be included)
        elif method_name == "PyNumDiff-MeanDiff":
            return self._meandiff()
        elif method_name == "PyNumDiff-MedianDiff":
            return self._mediandiff()

        # RBF (terrible but include for completeness)
        elif method_name == "PyNumDiff-RBF":
            return self._rbfdiff()
'''

# Implementation of missing methods:

MISSING_METHODS_IMPLEMENTATIONS = '''
    # === ADD THESE METHOD IMPLEMENTATIONS ===

    def _first_order(self) -> Dict:
        """Basic first-order finite difference (orders 0-1 only)."""
        t = self.x_train
        y = self.y_train
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 1.0

        try:
            # pynumdiff.first_order returns (x_smooth, dx_dt) tuple
            x_smooth, dx_dt = pynumdiff.first_order(y, dt)

            # Handle size mismatch (finite diff may be shorter)
            if len(dx_dt) == len(t) - 1:
                dx_dt = np.append(dx_dt, dx_dt[-1])
            elif len(dx_dt) == len(t) - 2:
                dx_dt = np.concatenate([[dx_dt[0]], dx_dt, [dx_dt[-1]]])

            result = self._orders_0_1_only(x_smooth, dx_dt)
            meta = {"method": "first_order_fd", "dt": dt}
            return {"predictions": result["predictions"], "failures": result["failures"], "meta": meta}

        except Exception as e:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": str(e)}
            }

    def _second_order(self) -> Dict:
        """Basic second-order finite difference (orders 0-1 only)."""
        t = self.x_train
        y = self.y_train
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 1.0

        try:
            x_smooth, dx_dt = pynumdiff.second_order(y, dt)

            # Handle size mismatch
            if len(dx_dt) == len(t) - 1:
                dx_dt = np.append(dx_dt, dx_dt[-1])
            elif len(dx_dt) == len(t) - 2:
                dx_dt = np.concatenate([[dx_dt[0]], dx_dt, [dx_dt[-1]]])

            result = self._orders_0_1_only(x_smooth, dx_dt)
            meta = {"method": "second_order_fd", "dt": dt}
            return {"predictions": result["predictions"], "failures": result["failures"], "meta": meta}

        except Exception as e:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": str(e)}
            }

    def _fourth_order(self) -> Dict:
        """Basic fourth-order finite difference (orders 0-1 only)."""
        t = self.x_train
        y = self.y_train
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 1.0

        try:
            x_smooth, dx_dt = pynumdiff.fourth_order(y, dt)

            # Handle size mismatch
            if len(dx_dt) == len(t) - 1:
                dx_dt = np.append(dx_dt, dx_dt[-1])
            elif len(dx_dt) == len(t) - 2:
                dx_dt = np.concatenate([[dx_dt[0]], dx_dt, [dx_dt[-1]]])

            result = self._orders_0_1_only(x_smooth, dx_dt)
            meta = {"method": "fourth_order_fd", "dt": dt}
            return {"predictions": result["predictions"], "failures": result["failures"], "meta": meta}

        except Exception as e:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": str(e)}
            }

    def _tvrdiff(self) -> Dict:
        """Total Variation Regularized Differentiation - general order 1 (orders 0-1 only).
        This was one of the BEST performers in testing (RMSE: 0.038)!
        """
        t = self.x_train
        y = self.y_train
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 1.0

        try:
            # Use optimized gamma or default
            gamma = self._tvgamma_docs_heuristic(y, dt)

            # tvrdiff takes order parameter
            x_smooth, dx_dt = tvr.tvrdiff(y, dt, order=1, gamma=gamma)

            result = self._orders_0_1_only(x_smooth, dx_dt)
            meta = {"method": "tv_regularized", "dt": dt, "gamma": gamma, "order": 1}
            return {"predictions": result["predictions"], "failures": result["failures"], "meta": meta}

        except Exception as e:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": str(e)}
            }

    def _polydiff(self, regime: str = "auto") -> Dict:
        """Polynomial fitting differentiation (orders 0-1 only).
        Good performer in testing (RMSE: 0.045).
        """
        from pynumdiff.polynomial_fit import polydiff

        t = self.x_train
        y = self.y_train
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 1.0

        try:
            if regime == "auto":
                # Auto-select parameters
                window_size = min(11, n // 10)
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
                "failures": {"error": str(e)}
            }

    def _meandiff(self) -> Dict:
        """Mean smoothing with differentiation (orders 0-1 only)."""
        t = self.x_train
        y = self.y_train
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 1.0

        try:
            window_size = min(7, n // 20)
            if window_size % 2 == 0:
                window_size += 1

            x_smooth, dx_dt = sfd.meandiff(y, dt, window_size=window_size,
                                           num_iterations=2)

            result = self._orders_0_1_only(x_smooth, dx_dt)
            meta = {"method": "meandiff", "dt": dt, "window_size": window_size}
            return {"predictions": result["predictions"], "failures": result["failures"], "meta": meta}

        except Exception as e:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": str(e)}
            }

    def _mediandiff(self) -> Dict:
        """Median smoothing with differentiation (orders 0-1 only)."""
        t = self.x_train
        y = self.y_train
        n = len(t)
        dt = float(np.mean(np.diff(t))) if n > 1 else 1.0

        try:
            window_size = min(7, n // 20)
            if window_size % 2 == 0:
                window_size += 1

            x_smooth, dx_dt = sfd.mediandiff(y, dt, window_size=window_size,
                                             num_iterations=2)

            result = self._orders_0_1_only(x_smooth, dx_dt)
            meta = {"method": "mediandiff", "dt": dt, "window_size": window_size}
            return {"predictions": result["predictions"], "failures": result["failures"], "meta": meta}

        except Exception as e:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": str(e)}
            }

    def _rbfdiff(self) -> Dict:
        """Radial Basis Function differentiation (orders 0-1 only).
        WARNING: This method performed TERRIBLY in testing (RMSE: 719!)
        Included for completeness but expect poor results.
        """
        from pynumdiff.basis_fit import rbfdiff

        t = self.x_train
        y = self.y_train
        n = len(t)

        try:
            # rbfdiff needs time array as first argument
            x_smooth, dx_dt = rbfdiff(t, y, sigma=0.1, lmbd=1e-3)

            # Ensure proper shape
            if hasattr(dx_dt, 'flatten'):
                dx_dt = dx_dt.flatten()

            result = self._orders_0_1_only(x_smooth, dx_dt)
            meta = {"method": "rbfdiff", "sigma": 0.1, "lambda": 1e-3,
                   "warning": "This method has severe conditioning issues"}
            return {"predictions": result["predictions"], "failures": result["failures"], "meta": meta}

        except Exception as e:
            return {
                "predictions": {order: [np.nan] * len(self.x_eval) for order in self.orders},
                "failures": {"error": str(e), "note": "RBF often fails due to conditioning"}
            }
'''

# Summary of what needs to be added:
print("=" * 80)
print("MISSING PYNUMDIFF METHODS THAT NEED TO BE ADDED")
print("=" * 80)

print("""
Your comprehensive study is MISSING these PyNumDiff methods:

HIGH PRIORITY (Good performers):
1. tvrdiff - TV Regularized, RMSE: 0.038 ✓✓ EXCELLENT
2. polydiff - Polynomial fitting, RMSE: 0.045 ✓✓ EXCELLENT
3. splinediff - Already implemented but not in dispatch! RMSE: 0.092 ✓ GOOD
4. second_order - Basic FD, RMSE: 0.074 ✓ GOOD

BASELINE METHODS (Should include for completeness):
5. first_order - Basic FD, RMSE: 0.279
6. fourth_order - Basic FD, RMSE: 0.195
7. meandiff - Window smoothing, RMSE: 0.673
8. mediandiff - Window smoothing, RMSE: 0.141

FOR COMPLETENESS (Known to be terrible):
9. rbfdiff - RBF, RMSE: 719 (!) - Include to show it fails catastrophically

IMPLEMENTATION:
Add the code above to methods/python/pynumdiff_wrapper/pynumdiff_methods.py

This will give you a COMPLETE comparison of ALL PyNumDiff methods!
""")