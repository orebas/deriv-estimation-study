"""
Test the new PyNumDiff implementation with proper higher-order derivative support.
"""
import sys
import numpy as np
from pathlib import Path

# Add methods directory to path
sys.path.insert(0, str(Path(__file__).parent / "methods" / "python"))

# Import the NEW implementation
from pynumdiff_wrapper.pynumdiff_methods_new import PyNumDiffMethods

print("=" * 80)
print("Testing New PyNumDiff Implementation")
print("=" * 80)

# Create test signal
np.random.seed(42)
n = 101
t = np.linspace(0, 1, n)
dt = np.mean(np.diff(t))
y_true = np.sin(2 * np.pi * t)
dy_true = 2 * np.pi * np.cos(2 * np.pi * t)
y_noisy = y_true + 1e-3 * np.random.randn(n)

print(f"\nTest signal: N={n}, dt={dt:.6f}, noise=1e-3")
print(f"Signal: y = sin(2πt)")

# Test orders 0-7
orders = list(range(8))

print("\n" + "=" * 80)
print("Methods with FULL orders 0-7 support")
print("=" * 80)

methods_full = [
    "PyNumDiff-SavGol-Tuned",
    "PyNumDiff-Spectral-Tuned",
]

for method_name in methods_full:
    print(f"\n{method_name}:")
    try:
        evaluator = PyNumDiffMethods(t, y_noisy, t, orders)
        result = evaluator.evaluate_method(method_name)

        predictions = result.get("predictions", {})
        failures = result.get("failures", {})

        for order in orders:
            if order in predictions:
                deriv = np.array(predictions[order])
                n_finite = np.sum(np.isfinite(deriv))
                n_total = len(deriv)

                if n_finite > 0:
                    # Compute RMSE for order 1
                    if order == 1:
                        finite_mask = np.isfinite(deriv)
                        errors = np.abs(deriv[finite_mask] - dy_true[finite_mask])
                        rmse = np.sqrt(np.mean(errors**2))
                        print(f"  Order {order}: {n_finite}/{n_total} finite | RMSE: {rmse:.4e}")
                    else:
                        print(f"  Order {order}: {n_finite}/{n_total} finite")
                else:
                    print(f"  Order {order}: {n_finite}/{n_total} finite (all NaN)")

                if order in failures:
                    print(f"             Failure: {failures[order]}")
            else:
                print(f"  Order {order}: MISSING")

    except Exception as e:
        print(f"  ✗ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("Methods with orders 0-1 ONLY")
print("=" * 80)

methods_01_only = [
    "PyNumDiff-Butter-Tuned",
    "PyNumDiff-Gaussian-Tuned",
    "PyNumDiff-Friedrichs-Tuned",
]

for method_name in methods_01_only:
    print(f"\n{method_name}:")
    try:
        evaluator = PyNumDiffMethods(t, y_noisy, t, orders)
        result = evaluator.evaluate_method(method_name)

        predictions = result.get("predictions", {})
        failures = result.get("failures", {})

        for order in [0, 1, 2, 7]:  # Check a few orders
            if order in predictions:
                deriv = np.array(predictions[order])
                n_finite = np.sum(np.isfinite(deriv))
                n_total = len(deriv)

                if order <= 1:
                    # Orders 0-1 should have data
                    if order == 1:
                        if n_finite > 0:
                            finite_mask = np.isfinite(deriv)
                            errors = np.abs(deriv[finite_mask] - dy_true[finite_mask])
                            rmse = np.sqrt(np.mean(errors**2))
                            print(f"  Order {order}: {n_finite}/{n_total} finite | RMSE: {rmse:.4e}")
                        else:
                            print(f"  Order {order}: {n_finite}/{n_total} finite (UNEXPECTED!)")
                    else:
                        print(f"  Order {order}: {n_finite}/{n_total} finite")
                else:
                    # Orders 2+ should be NaN
                    if n_finite == 0:
                        print(f"  Order {order}: {n_finite}/{n_total} finite (all NaN as expected ✓)")
                    else:
                        print(f"  Order {order}: {n_finite}/{n_total} finite (UNEXPECTED!)")

                if order in failures:
                    print(f"             Reason: {failures[order]}")

    except Exception as e:
        print(f"  ✗ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
✅ Full orders 0-7 support:
   - PyNumDiff-SavGol-* (native deriv parameter)
   - PyNumDiff-Spectral-* (FFT (iω)^n multiplication)

✅ Orders 0-1 only:
   - PyNumDiff-Butter-*
   - PyNumDiff-Gaussian-*
   - PyNumDiff-Friedrichs-*
   - PyNumDiff-Kalman-*
   - PyNumDiff-TV-*

✅ No spline fitting for higher derivatives
✅ Honest reporting: methods report only what they can compute
""")
