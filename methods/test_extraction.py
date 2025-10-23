#!/usr/bin/env python3
"""
Test script to validate extracted methods match original behavior.

Usage:
    python test_extraction.py <category> <method_name>

Examples:
    python test_extraction.py gp gp_rbf_mean
    python test_extraction.py splines chebyshev
"""

import sys
import numpy as np
from pathlib import Path

# Add paths to import both old and new implementations
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).parent / "python"))

def create_test_data():
    """Create standard test data for validation."""
    np.random.seed(42)
    x_train = np.linspace(0, 1, 50)
    y_train = np.sin(2 * np.pi * x_train) + 0.01 * np.random.randn(50)
    x_eval = np.linspace(0, 1, 20)
    orders = [0, 1, 2, 3]
    return x_train, y_train, x_eval, orders


def compare_results(result1, result2, method_name, rtol=1e-6, atol=1e-5):
    """
    Compare two result dictionaries for equivalence.

    Args:
        result1: Result from original implementation
        result2: Result from extracted implementation
        method_name: Name of method being tested
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        bool: True if results match, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Testing: {method_name}")
    print(f"{'='*60}")

    # Check structure
    if "predictions" not in result1 or "predictions" not in result2:
        print("❌ FAIL: Missing 'predictions' field")
        return False

    preds1 = result1["predictions"]
    preds2 = result2["predictions"]

    # Check orders match
    if set(preds1.keys()) != set(preds2.keys()):
        print(f"❌ FAIL: Order mismatch")
        print(f"  Original orders: {sorted(preds1.keys())}")
        print(f"  Extracted orders: {sorted(preds2.keys())}")
        return False

    # Check each order
    all_match = True
    for order in sorted(preds1.keys()):
        arr1 = np.array(preds1[order])
        arr2 = np.array(preds2[order])

        # Check lengths
        if len(arr1) != len(arr2):
            print(f"❌ FAIL Order {order}: Length mismatch ({len(arr1)} vs {len(arr2)})")
            all_match = False
            continue

        # Handle NaN values
        nan_mask1 = np.isnan(arr1)
        nan_mask2 = np.isnan(arr2)

        if not np.array_equal(nan_mask1, nan_mask2):
            print(f"❌ FAIL Order {order}: NaN pattern mismatch")
            all_match = False
            continue

        # Compare finite values
        valid_mask = ~nan_mask1 & ~nan_mask2
        if np.any(valid_mask):
            valid1 = arr1[valid_mask]
            valid2 = arr2[valid_mask]

            if not np.allclose(valid1, valid2, rtol=rtol, atol=atol):
                max_diff = np.max(np.abs(valid1 - valid2))
                rel_diff = max_diff / (np.max(np.abs(valid1)) + 1e-12)
                print(f"❌ FAIL Order {order}:")
                print(f"  Max absolute difference: {max_diff:.2e}")
                print(f"  Max relative difference: {rel_diff:.2e}")
                print(f"  Sample original: {valid1[:3]}")
                print(f"  Sample extracted: {valid2[:3]}")
                all_match = False
            else:
                print(f"✅ PASS Order {order}: Predictions match (rtol={rtol:.1e}, atol={atol:.1e})")

    return all_match


def test_gp_methods():
    """Test Gaussian Process methods."""
    from python_methods import MethodEvaluator as OriginalEvaluator
    from python.gp.gaussian_process import GPMethods

    x_train, y_train, x_eval, orders = create_test_data()

    # Test GP RBF
    print("\n" + "="*60)
    print("TESTING: GP RBF Method")
    print("="*60)

    orig = OriginalEvaluator(x_train, y_train, x_eval, orders)
    extracted = GPMethods(x_train, y_train, x_eval, orders)

    result_orig = orig.evaluate_method("gp_rbf_mean")
    result_extracted = extracted.evaluate_method("gp_rbf_mean")

    if compare_results(result_orig, result_extracted, "GP RBF"):
        print("\n✅ GP RBF: ALL TESTS PASSED")
        return True
    else:
        print("\n❌ GP RBF: TESTS FAILED")
        return False


def test_splines_methods():
    """Test Splines methods."""
    from python_methods import MethodEvaluator as OriginalEvaluator
    from python.splines.splines import SplineMethods

    x_train, y_train, x_eval, orders = create_test_data()

    # Test all spline methods
    spline_methods = [
        ("chebyshev", "Chebyshev"),
        ("RKHS_Spline_m2_Python", "RKHS Spline m=2"),
        ("Butterworth_Python", "Butterworth"),
        ("ButterworthSpline_Python", "FiniteDiff+Spline"),
        ("SVR_Python", "SVR+Spline"),
    ]

    all_passed = True
    for method_name, display_name in spline_methods:
        print(f"\n{'='*60}")
        print(f"TESTING: {display_name} Method")
        print(f"{'='*60}")

        orig = OriginalEvaluator(x_train, y_train, x_eval, orders)
        extracted = SplineMethods(x_train, y_train, x_eval, orders)

        try:
            result_orig = orig.evaluate_method(method_name)
            result_extracted = extracted.evaluate_method(method_name)

            if compare_results(result_orig, result_extracted, display_name):
                print(f"✅ {display_name}: PASSED")
            else:
                print(f"❌ {display_name}: FAILED")
                all_passed = False
        except Exception as e:
            print(f"❌ {display_name}: EXCEPTION - {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    return all_passed


def test_method(category, method_name):
    """
    Test a specific method from a category.

    Args:
        category: Category name (e.g., 'gp', 'splines')
        method_name: Method identifier for evaluate_method()
    """
    from python_methods import MethodEvaluator as OriginalEvaluator

    # Import extracted category
    if category == "gp":
        from python.gp.gaussian_process import GPMethods as CategoryMethods
    elif category == "splines":
        from python.splines.splines import SplineMethods as CategoryMethods
    elif category == "spectral":
        from python.spectral.spectral import SpectralMethods as CategoryMethods
    elif category == "adaptive":
        from python.adaptive.adaptive import AdaptiveMethods as CategoryMethods
    elif category == "filtering":
        from python.filtering.filters import FilteringMethods as CategoryMethods
    else:
        print(f"❌ Unknown category: {category}")
        return False

    x_train, y_train, x_eval, orders = create_test_data()

    orig = OriginalEvaluator(x_train, y_train, x_eval, orders)
    extracted = CategoryMethods(x_train, y_train, x_eval, orders)

    try:
        result_orig = orig.evaluate_method(method_name)
        result_extracted = extracted.evaluate_method(method_name)

        return compare_results(result_orig, result_extracted, method_name)

    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_filtering_methods():
    """Test Filtering methods."""
    from python_methods import MethodEvaluator as OriginalEvaluator
    from python.filtering.filters import FilteringMethods

    x_train, y_train, x_eval, orders = create_test_data()

    # Test all filtering methods (except TVRegDiff which may not be available)
    filtering_methods = [
        ("Whittaker_m2_Python", "Whittaker m=2"),
        ("SavitzkyGolay_Python", "Savitzky-Golay"),
        ("KalmanGrad_Python", "Kalman RTS"),
    ]

    all_passed = True
    for method_name, display_name in filtering_methods:
        print(f"\n{'='*60}")
        print(f"TESTING: {display_name} Method")
        print(f"{'='*60}")

        orig = OriginalEvaluator(x_train, y_train, x_eval, orders)
        extracted = FilteringMethods(x_train, y_train, x_eval, orders)

        try:
            result_orig = orig.evaluate_method(method_name)
            result_extracted = extracted.evaluate_method(method_name)

            if compare_results(result_orig, result_extracted, display_name):
                print(f"✅ {display_name}: PASSED")
            else:
                print(f"❌ {display_name}: FAILED")
                all_passed = False
        except Exception as e:
            print(f"❌ {display_name}: EXCEPTION - {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    return all_passed


def test_adaptive_methods():
    """Test Adaptive AAA methods."""
    from python_methods import MethodEvaluator as OriginalEvaluator
    from python.adaptive.adaptive import AdaptiveMethods

    x_train, y_train, x_eval, orders = create_test_data()

    # Test Python AAA methods (baryrat-based)
    # Note: JAX methods may fail if JAX not available or have different behavior
    adaptive_methods = [
        ("AAA-Python-Adaptive-Wavelet", "AAA Wavelet"),
        ("AAA-Python-Adaptive-Diff2", "AAA Diff2"),
    ]

    all_passed = True
    for method_name, display_name in adaptive_methods:
        print(f"\n{'='*60}")
        print(f"TESTING: {display_name} Method")
        print(f"{'='*60}")

        orig = OriginalEvaluator(x_train, y_train, x_eval, orders)
        extracted = AdaptiveMethods(x_train, y_train, x_eval, orders)

        try:
            result_orig = orig.evaluate_method(method_name)
            result_extracted = extracted.evaluate_method(method_name)

            if compare_results(result_orig, result_extracted, display_name):
                print(f"✅ {display_name}: PASSED")
            else:
                print(f"❌ {display_name}: FAILED")
                all_passed = False
        except Exception as e:
            print(f"❌ {display_name}: EXCEPTION - {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    return all_passed


def test_spectral_methods():
    """Test Spectral methods."""
    from python_methods import MethodEvaluator as OriginalEvaluator
    from python.spectral.spectral import SpectralMethods

    x_train, y_train, x_eval, orders = create_test_data()

    # Test spectral methods that don't require special dependencies
    spectral_methods = [
        ("fourier", "Fourier"),
        ("fourier_continuation", "Fourier Continuation"),
    ]

    all_passed = True
    for method_name, display_name in spectral_methods:
        print(f"\n{'='*60}")
        print(f"TESTING: {display_name} Method")
        print(f"{'='*60}")

        orig = OriginalEvaluator(x_train, y_train, x_eval, orders)
        extracted = SpectralMethods(x_train, y_train, x_eval, orders)

        try:
            result_orig = orig.evaluate_method(method_name)
            result_extracted = extracted.evaluate_method(method_name)

            if compare_results(result_orig, result_extracted, display_name):
                print(f"✅ {display_name}: PASSED")
            else:
                print(f"❌ {display_name}: FAILED")
                all_passed = False
        except Exception as e:
            print(f"❌ {display_name}: EXCEPTION - {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    return all_passed


def main():
    """Main test runner."""
    if len(sys.argv) < 2:
        print("Running all available tests...\n")
        # Run GP tests
        gp_success = test_gp_methods()
        # Run Splines tests
        splines_success = test_splines_methods()
        # Run Filtering tests
        filtering_success = test_filtering_methods()
        # Run Adaptive tests
        adaptive_success = test_adaptive_methods()
        # Run Spectral tests
        spectral_success = test_spectral_methods()

        overall_success = gp_success and splines_success and filtering_success and adaptive_success and spectral_success
        print(f"\n{'='*60}")
        print("OVERALL RESULTS:")
        print(f"  GP Methods: {'✅ PASSED' if gp_success else '❌ FAILED'}")
        print(f"  Spline Methods: {'✅ PASSED' if splines_success else '❌ FAILED'}")
        print(f"  Filtering Methods: {'✅ PASSED' if filtering_success else '❌ FAILED'}")
        print(f"  Adaptive Methods: {'✅ PASSED' if adaptive_success else '❌ FAILED'}")
        print(f"  Spectral Methods: {'✅ PASSED' if spectral_success else '❌ FAILED'}")
        print(f"{'='*60}")
        sys.exit(0 if overall_success else 1)

    elif len(sys.argv) == 3:
        category = sys.argv[1]
        method_name = sys.argv[2]
        success = test_method(category, method_name)
        sys.exit(0 if success else 1)

    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
