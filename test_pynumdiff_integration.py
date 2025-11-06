"""
Quick integration test for PyNumDiff methods.

Verifies:
1. Methods can be imported and instantiated
2. Methods can be called without errors
3. Results have correct structure and reasonable values
"""

import sys
import numpy as np
from pathlib import Path

# Add methods directory to path
sys.path.insert(0, str(Path(__file__).parent / "methods" / "python"))

from pynumdiff_wrapper.pynumdiff_methods import PyNumDiffMethods

def test_pynumdiff_integration():
    """Run a simple integration test."""

    print("=" * 80)
    print("PyNumDiff Integration Test")
    print("=" * 80)

    # Create test signal: smooth ODE-like trajectory
    np.random.seed(42)
    n = 101
    x = np.linspace(0, 1, n)
    dx = np.mean(np.diff(x))

    # Test signal: y = sin(2πx) + noise
    y_true = np.sin(2 * np.pi * x)
    dy_true = 2 * np.pi * np.cos(2 * np.pi * x)

    # Add moderate noise
    noise_level = 1e-3
    y_noisy = y_true + noise_level * np.random.randn(n)

    # Evaluation points (same as training for simplicity)
    x_eval = x

    # Derivative orders to test
    orders = [0, 1, 2, 3]

    print(f"\nTest Setup:")
    print(f"  N = {n}")
    print(f"  dx = {dx:.6f}")
    print(f"  Noise level = {noise_level}")
    print(f"  Signal: y = sin(2πx)")
    print()

    # Test a subset of methods (one from each family)
    test_methods = [
        "PyNumDiff-Butter-Auto",
        "PyNumDiff-Spline-Tuned",
        "PyNumDiff-Gaussian-Auto",
        "PyNumDiff-Kalman-Auto",
        "PyNumDiff-TV-Velocity",
    ]

    print("Testing Methods:")
    print("-" * 80)

    success_count = 0
    total_count = len(test_methods)

    for method_name in test_methods:
        print(f"\n{method_name}:")

        try:
            # Create evaluator
            evaluator = PyNumDiffMethods(x, y_noisy, x_eval, orders)

            # Evaluate method
            result = evaluator.evaluate_method(method_name)

            # Check result structure
            if "predictions" not in result:
                print(f"  ✗ ERROR: Missing 'predictions' key")
                continue

            predictions = result["predictions"]
            failures = result.get("failures", {})

            # Check each order
            all_valid = True
            for order in orders:
                if order not in predictions:
                    print(f"  ✗ Order {order}: MISSING")
                    all_valid = False
                    continue

                deriv = predictions[order]

                # Check if result is valid
                if not isinstance(deriv, (list, np.ndarray)):
                    print(f"  ✗ Order {order}: Invalid type {type(deriv)}")
                    all_valid = False
                    continue

                deriv_arr = np.array(deriv)
                n_finite = np.sum(np.isfinite(deriv_arr))
                n_total = len(deriv_arr)
                pct_finite = 100 * n_finite / n_total

                print(f"  Order {order}: {n_finite}/{n_total} finite ({pct_finite:.1f}%)", end="")

                # For order 1, compute error against true derivative
                if order == 1 and n_finite > 0:
                    finite_mask = np.isfinite(deriv_arr)
                    errors = np.abs(deriv_arr[finite_mask] - dy_true[finite_mask])
                    rmse = np.sqrt(np.mean(errors**2))
                    print(f" | RMSE: {rmse:.4e}", end="")

                print()  # newline

            if all_valid:
                print(f"  ✓ SUCCESS")
                success_count += 1
            else:
                print(f"  ✗ PARTIAL SUCCESS (some orders failed)")

        except Exception as e:
            print(f"  ✗ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"\nPassed: {success_count}/{total_count} methods")

    if success_count == total_count:
        print("\n✅ All tests passed! PyNumDiff integration is working.")
        return 0
    else:
        print(f"\n⚠️  {total_count - success_count} method(s) failed. Review errors above.")
        return 1

if __name__ == "__main__":
    exit_code = test_pynumdiff_integration()
    sys.exit(exit_code)
