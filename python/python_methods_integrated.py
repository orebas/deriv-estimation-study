"""
Python-based Differentiation Methods (Integrated from Extracted Modules)

This script integrates all extracted Python methods from methods/python/* into the
main benchmark pipeline, replacing the monolithic python_methods.py.

Usage:
    python python_methods_integrated.py <input_json> <output_json>
"""

import sys
import json
import os
import time
from pathlib import Path
from typing import Dict, List
import numpy as np

# Add methods directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "methods" / "python"))

# Import extracted method modules
try:
    from gp.gaussian_process import GPMethods
    from splines.splines import SplineMethods  # Fixed: SplineMethods not SplinesMethods
    from filtering.filters import FilteringMethods
    from adaptive.adaptive import AdaptiveMethods
    from spectral.spectral import SpectralMethods
    EXTRACTED_METHODS_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: Failed to import extracted methods: {e}")
    print(f"sys.path: {sys.path}")
    sys.exit(1)


def _safe_orders(data_orders: List[int]) -> List[int]:
    """Safely parse derivative orders."""
    if data_orders is None:
        return list(range(8))
    try:
        return [int(o) for o in data_orders]
    except Exception:
        return list(range(8))


def _validate_input_data(data: dict):
    """Validate input JSON data."""
    required = ["times", "y_noisy"]
    for key in required:
        if key not in data and (key.replace("_", "") not in data):
            raise ValueError(f"Missing required field: {key}")


class IntegratedMethodEvaluator:
    """Evaluates Python-based differentiation methods using extracted modules."""

    def __init__(self, x_train, y_train, x_eval, orders):
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)
        self.x_eval = np.array(x_eval)
        self.orders = orders

        # Input validation
        if not np.all(np.isfinite(self.x_train)):
            raise ValueError("Training x values contain NaN or inf")
        if not np.all(np.isfinite(self.y_train)):
            raise ValueError("Training y values contain NaN or inf")
        if not np.all(np.isfinite(self.x_eval)):
            raise ValueError("Evaluation x values contain NaN or inf")

        # Initialize category evaluators (lazy initialization to avoid overhead)
        self._gp_eval = None
        self._splines_eval = None
        self._filtering_eval = None
        self._adaptive_eval = None
        self._spectral_eval = None

    def _get_gp_evaluator(self):
        """Get or create GP evaluator."""
        if self._gp_eval is None:
            self._gp_eval = GPMethods(self.x_train, self.y_train, self.x_eval, self.orders)
        return self._gp_eval

    def _get_splines_evaluator(self):
        """Get or create Splines evaluator."""
        if self._splines_eval is None:
            self._splines_eval = SplineMethods(self.x_train, self.y_train, self.x_eval, self.orders)
        return self._splines_eval

    def _get_filtering_evaluator(self):
        """Get or create Filtering evaluator."""
        if self._filtering_eval is None:
            self._filtering_eval = FilteringMethods(self.x_train, self.y_train, self.x_eval, self.orders)
        return self._filtering_eval

    def _get_adaptive_evaluator(self):
        """Get or create Adaptive evaluator."""
        if self._adaptive_eval is None:
            self._adaptive_eval = AdaptiveMethods(self.x_train, self.y_train, self.x_eval, self.orders)
        return self._adaptive_eval

    def _get_spectral_evaluator(self):
        """Get or create Spectral evaluator."""
        if self._spectral_eval is None:
            self._spectral_eval = SpectralMethods(self.x_train, self.y_train, self.x_eval, self.orders)
        return self._spectral_eval

    def evaluate_method(self, method_name: str) -> Dict:
        """
        Evaluate a single method by name.

        Maps method names to extracted module functions.
        """
        t_start = time.time()

        try:
            # Map method names to evaluator getters (pass public name to evaluator)
            method_map = {
                # GP methods
                "gp_rbf_mean": self._get_gp_evaluator,
                "GP_RBF_Python": self._get_gp_evaluator,
                "GP_RBF_Iso_Python": self._get_gp_evaluator,
                "GP_Matern_Python": self._get_gp_evaluator,
                "GP_Matern_1.5_Python": self._get_gp_evaluator,
                "GP_Matern_2.5_Python": self._get_gp_evaluator,

                # Splines methods
                "chebyshev": self._get_splines_evaluator,
                "Chebyshev-AICc": self._get_splines_evaluator,
                "RKHS_Spline_m2_Python": self._get_splines_evaluator,
                "Butterworth_Python": self._get_splines_evaluator,
                "ButterworthSpline_Python": self._get_splines_evaluator,
                "SVR_Python": self._get_splines_evaluator,

                # Filtering methods
                "Whittaker_m2_Python": self._get_filtering_evaluator,
                "SavitzkyGolay_Python": self._get_filtering_evaluator,
                "KalmanGrad_Python": self._get_filtering_evaluator,
                "TVRegDiff_Python": self._get_filtering_evaluator,

                # Adaptive methods
                "AAA-Python-Adaptive-Wavelet": self._get_adaptive_evaluator,
                "AAA-Python-Adaptive-Diff2": self._get_adaptive_evaluator,
                "AAA-JAX-Adaptive-Wavelet": self._get_adaptive_evaluator,
                "AAA-JAX-Adaptive-Diff2": self._get_adaptive_evaluator,

                # Spectral methods
                "fourier": self._get_spectral_evaluator,
                "Fourier-GCV": self._get_spectral_evaluator,
                "Fourier-FFT-Adaptive": self._get_spectral_evaluator,
                "fourier_continuation": self._get_spectral_evaluator,
                "Fourier-Continuation-Adaptive": self._get_spectral_evaluator,
                "ad_trig": self._get_spectral_evaluator,
                "ad_trig_adaptive": self._get_spectral_evaluator,
                "SpectralTaper_Python": self._get_spectral_evaluator,
            }

            if method_name not in method_map:
                return {
                    "success": False,
                    "error": f"Unknown method: {method_name}",
                    "predictions": {},
                    "failures": {0: f"Unknown method: {method_name}"},
                    "timing": time.time() - t_start,
                    "meta": {}
                }

            # Get evaluator and call method with public method name
            get_evaluator = method_map[method_name]
            evaluator = get_evaluator()
            result = evaluator.evaluate_method(method_name)

            # Add timing
            result["timing"] = time.time() - t_start
            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "predictions": {},
                "failures": {0: str(e)},
                "timing": time.time() - t_start,
                "meta": {}
            }


def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: python python_methods_integrated.py <input_json> <output_json>")
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

    # Parse input
    x_train = data.get("t") or data.get("times")
    y_train = data.get("y") or data.get("y_noisy")
    x_eval = data.get("t_eval") or x_train
    orders = _safe_orders(data.get("orders"))

    # Environment overrides
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

    print(f"Processing {input_file.name} (using EXTRACTED methods)...")
    print(f"  Data points: {len(x_train)}")
    print(f"  Orders: {orders}")

    # Methods to evaluate (matching original list)
    methods = [
        # Analytic/closed-form
        "chebyshev",
        "fourier",
        "fourier_continuation",
        "gp_rbf_mean",
        "ad_trig",
        # Adaptive hyperparameter methods
        "Chebyshev-AICc",
        "Fourier-GCV",
        "Fourier-FFT-Adaptive",
        "Fourier-Continuation-Adaptive",
        "ad_trig_adaptive",
        "AAA-Python-Adaptive-Wavelet",
        "AAA-Python-Adaptive-Diff2",
        "AAA-JAX-Adaptive-Wavelet",
        "AAA-JAX-Adaptive-Diff2",
        # IFAC25 legacy methods
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
    include_matern = os.environ.get("PY_INCLUDE_MATERN", "0").lower() not in ("0", "false", "no")
    if not include_matern:
        methods = [m for m in methods if not m.startswith("GP_Matern")]

    exclude_methods_csv = os.environ.get("PY_EXCLUDE", "").strip()
    if exclude_methods_csv:
        excludes = {m.strip() for m in exclude_methods_csv.split(",") if m.strip()}
        methods = [m for m in methods if m not in excludes]

    only_methods_csv = os.environ.get("PY_METHODS", "").strip()
    if only_methods_csv:
        onlys = [m.strip() for m in only_methods_csv.split(",") if m.strip()]
        if onlys:
            methods = onlys

    # Evaluate methods
    results = {}
    evaluator = IntegratedMethodEvaluator(x_train, y_train, x_eval, orders)

    for method in methods:
        print(f"  Evaluating {method}...")
        results[method] = evaluator.evaluate_method(method)

    # Write output (matching original format)
    def _clean_predictions(preds: dict, method_name: str = "Unknown") -> dict:
        cleaned = {}
        for k, vals in preds.items():
            try:
                arr = np.asarray(vals, dtype=float)
                if arr.size == 0:
                    continue
                if not np.all(np.isfinite(arr)):
                    print(f"    WARNING: Non-finite values in {method_name} order {k}, data excluded")
                    continue
                cleaned[str(k)] = [float(x) for x in arr]
            except Exception:
                continue
        return cleaned

    cleaned_results = {}
    for m, res in results.items():
        new_res = dict(res)
        new_res["predictions"] = _clean_predictions(res.get("predictions", {}), m)
        # Infer success from presence of valid predictions
        new_res["success"] = len(new_res["predictions"]) > 0 and len(res.get("failures", {})) == 0
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
    for method, result in cleaned_results.items():
        if result.get("success", False):
            valid_orders = sum(1 for o in orders if str(o) in result.get("predictions", {})
                             and not any(np.isnan(result["predictions"][str(o)])))
            print(f"    {method}: OK ({valid_orders}/{len(orders)} orders, "
                  f"{result.get('timing', 0.0):.3f}s)")
        else:
            print(f"    {method}: FAILED - {result.get('error', 'Unknown')}")


if __name__ == "__main__":
    main()
