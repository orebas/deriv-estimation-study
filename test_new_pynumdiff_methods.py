"""
Test script to verify that all newly integrated PyNumDiff methods are working
"""

import numpy as np
import sys
from pathlib import Path

# Add the wrapper directory to path
sys.path.insert(0, str(Path(__file__).parent / 'methods' / 'python' / 'pynumdiff_wrapper'))

from pynumdiff_methods import PyNumDiffMethods

def test_new_methods():
    """Test all newly integrated PyNumDiff methods"""

    # Create test signal: x^(5/2) + sin(2x) with noise
    t = np.linspace(0.1, 3, 251)
    dt = t[1] - t[0]

    # True signal and derivatives
    y_true = t**(5/2) + np.sin(2*t)
    dy_true = 2.5 * t**(3/2) + 2*np.cos(2*t)

    # Add moderate noise
    np.random.seed(42)
    noise_level = 0.01
    y_noisy = y_true + noise_level * np.random.randn(len(t))

    # Initialize evaluator
    evaluator = PyNumDiffMethods(
        x_train=t,
        y_train=y_noisy,
        x_eval=t,
        orders=[0, 1]
    )

    # List of new methods to test
    new_methods = [
        # TV Regularized (BEST performers)
        ("PyNumDiff-TVRegularized-Auto", "tvrdiff (auto)", True),
        ("PyNumDiff-TVRegularized-Tuned", "tvrdiff (tuned)", True),

        # Polynomial fitting (EXCELLENT)
        ("PyNumDiff-PolyDiff-Auto", "polydiff (auto)", True),
        ("PyNumDiff-PolyDiff-Tuned", "polydiff (tuned)", True),

        # Basic finite differences
        ("PyNumDiff-FirstOrder", "first_order FD", True),
        ("PyNumDiff-SecondOrder", "second_order FD", True),
        ("PyNumDiff-FourthOrder", "fourth_order FD", True),

        # Window-based methods
        ("PyNumDiff-MeanDiff-Auto", "meandiff (auto)", True),
        ("PyNumDiff-MeanDiff-Tuned", "meandiff (tuned)", True),
        ("PyNumDiff-MedianDiff-Auto", "mediandiff (auto)", True),
        ("PyNumDiff-MedianDiff-Tuned", "mediandiff (tuned)", True),

        # RBF (expected to fail)
        ("PyNumDiff-RBF-Auto", "rbfdiff (auto)", False),
        ("PyNumDiff-RBF-Tuned", "rbfdiff (tuned)", False),

        # Also test that Spline dispatch works
        ("PyNumDiff-Spline-Auto", "splinediff (auto)", True),
        ("PyNumDiff-Spline-Tuned", "splinediff (tuned)", True),
    ]

    print("=" * 80)
    print("TESTING NEWLY INTEGRATED PYNUMDIFF METHODS")
    print("=" * 80)
    print()

    successes = 0
    failures = 0

    for method_name, display_name, expect_success in new_methods:
        print(f"\nTesting {display_name} ({method_name})...")
        try:
            result = evaluator.evaluate_method(method_name)

            # Check if method returned valid results
            if "predictions" not in result:
                print(f"  ❌ FAIL: No predictions returned")
                failures += 1
                continue

            # Check for explicit failures (except for RBF where we expect them)
            if "failures" in result and result["failures"]:
                if not expect_success:
                    print(f"  ⚠️  Expected failure: {result['failures']}")
                    successes += 1
                else:
                    print(f"  ❌ FAIL: {result['failures']}")
                    failures += 1
                continue

            # Get predictions
            preds = result["predictions"]

            # Check that we have order 0 and 1
            if 0 not in preds or 1 not in preds:
                print(f"  ❌ FAIL: Missing order 0 or 1 in predictions")
                failures += 1
                continue

            # Check for NaNs
            y_smooth = np.array(preds[0])
            dy_dt = np.array(preds[1])

            if np.all(np.isnan(dy_dt)):
                if not expect_success:
                    print(f"  ⚠️  Expected to fail (all NaN)")
                    successes += 1
                else:
                    print(f"  ❌ FAIL: All derivatives are NaN")
                    failures += 1
                continue

            # Calculate RMSE for derivative
            valid_mask = ~np.isnan(dy_dt)
            if np.sum(valid_mask) < 10:
                print(f"  ❌ FAIL: Too few valid points ({np.sum(valid_mask)})")
                failures += 1
                continue

            rmse = np.sqrt(np.mean((dy_dt[valid_mask] - dy_true[valid_mask])**2))

            # Categorize performance
            if rmse < 0.05:
                category = "EXCELLENT"
                symbol = "✅✅"
            elif rmse < 0.1:
                category = "GOOD"
                symbol = "✅"
            elif rmse < 0.5:
                category = "OK"
                symbol = "✓"
            elif rmse < 5.0:
                category = "POOR"
                symbol = "⚠️"
            else:
                category = "TERRIBLE"
                symbol = "❌"

            print(f"  {symbol} {category}: RMSE = {rmse:.4f}")

            # Show metadata if available
            if "meta" in result:
                meta = result["meta"]
                if "warning" in meta:
                    print(f"     Warning: {meta['warning']}")
                if "gamma" in meta:
                    print(f"     Parameters: gamma={meta['gamma']:.1e}")
                elif "window_size" in meta:
                    print(f"     Parameters: window={meta.get('window_size', 'N/A')}, degree={meta.get('degree', 'N/A')}")

            successes += 1

        except Exception as e:
            print(f"  ❌ EXCEPTION: {str(e)}")
            failures += 1

    print("\n" + "=" * 80)
    print(f"SUMMARY: {successes} successes, {failures} failures out of {len(new_methods)} methods")
    print("=" * 80)

    # Test that we can list all methods
    print("\n\nAll available PyNumDiff methods:")
    print("-" * 40)

    all_methods = [
        "PyNumDiff-SavGol-Auto", "PyNumDiff-SavGol-Tuned",
        "PyNumDiff-Spectral-Auto", "PyNumDiff-Spectral-Tuned",
        "PyNumDiff-Butter-Auto", "PyNumDiff-Butter-Tuned",
        "PyNumDiff-Spline-Auto", "PyNumDiff-Spline-Tuned",
        "PyNumDiff-Gaussian-Auto", "PyNumDiff-Gaussian-Tuned",
        "PyNumDiff-Friedrichs-Auto", "PyNumDiff-Friedrichs-Tuned",
        "PyNumDiff-Kalman-Auto", "PyNumDiff-Kalman-Tuned",
        "PyNumDiff-TV-Velocity", "PyNumDiff-TV-Acceleration", "PyNumDiff-TV-Jerk",
        "PyNumDiff-TVRegularized-Auto", "PyNumDiff-TVRegularized-Tuned",
        "PyNumDiff-PolyDiff-Auto", "PyNumDiff-PolyDiff-Tuned",
        "PyNumDiff-FirstOrder", "PyNumDiff-SecondOrder", "PyNumDiff-FourthOrder",
        "PyNumDiff-MeanDiff-Auto", "PyNumDiff-MeanDiff-Tuned",
        "PyNumDiff-MedianDiff-Auto", "PyNumDiff-MedianDiff-Tuned",
        "PyNumDiff-RBF-Auto", "PyNumDiff-RBF-Tuned",
    ]

    print(f"Total methods available: {len(all_methods)}")
    print("\nMethods by category:")
    print("  Full orders 0-7: 4 methods")
    print("  Orders 0-1 only: {len(all_methods) - 4} methods")
    print("  TOTAL: {len(all_methods)} methods!")

if __name__ == "__main__":
    test_new_methods()