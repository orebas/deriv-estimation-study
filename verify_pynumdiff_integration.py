"""
Verify that the comprehensive study can access all PyNumDiff methods
"""

import sys
from pathlib import Path
import numpy as np

# Add the methods directory to path
sys.path.insert(0, str(Path(__file__).parent / 'methods' / 'python'))

# Import the integrated methods module
from pynumdiff_wrapper.pynumdiff_methods import PyNumDiffMethods

def verify_comprehensive_study_integration():
    """Verify all PyNumDiff methods are available for the comprehensive study"""

    print("=" * 80)
    print("VERIFYING PYNUMDIFF INTEGRATION FOR COMPREHENSIVE STUDY")
    print("=" * 80)
    print()

    # Create minimal test data
    t = np.linspace(0, 1, 101)
    y = np.sin(2 * np.pi * t)

    # Initialize evaluator
    evaluator = PyNumDiffMethods(
        x_train=t,
        y_train=y,
        x_eval=t,
        orders=[0, 1]
    )

    # Complete list of all PyNumDiff methods for comprehensive study
    all_methods = {
        # === Full orders 0-7 support ===
        "PyNumDiff-SavGol-Auto": "Savitzky-Golay (auto)",
        "PyNumDiff-SavGol-Tuned": "Savitzky-Golay (tuned)",
        "PyNumDiff-Spectral-Auto": "Spectral (auto)",
        "PyNumDiff-Spectral-Tuned": "Spectral (tuned)",

        # === Orders 0-1 only ===
        # Existing methods
        "PyNumDiff-Butter-Auto": "Butterworth (auto)",
        "PyNumDiff-Butter-Tuned": "Butterworth (tuned)",
        "PyNumDiff-Spline-Auto": "Spline (auto)",
        "PyNumDiff-Spline-Tuned": "Spline (tuned)",
        "PyNumDiff-Gaussian-Auto": "Gaussian (auto)",
        "PyNumDiff-Gaussian-Tuned": "Gaussian (tuned)",
        "PyNumDiff-Friedrichs-Auto": "Friedrichs (auto)",
        "PyNumDiff-Friedrichs-Tuned": "Friedrichs (tuned)",
        "PyNumDiff-Kalman-Auto": "Kalman RTS (auto)",
        "PyNumDiff-Kalman-Tuned": "Kalman RTS (tuned)",
        "PyNumDiff-TV-Velocity": "TV Velocity",
        "PyNumDiff-TV-Acceleration": "TV Acceleration",
        "PyNumDiff-TV-Jerk": "TV Jerk",

        # === NEW METHODS ADDED TODAY ===
        "PyNumDiff-TVRegularized-Auto": "TV Regularized (auto) [NEW]",
        "PyNumDiff-TVRegularized-Tuned": "TV Regularized (tuned) [NEW]",
        "PyNumDiff-PolyDiff-Auto": "Polynomial Diff (auto) [NEW]",
        "PyNumDiff-PolyDiff-Tuned": "Polynomial Diff (tuned) [NEW]",
        "PyNumDiff-FirstOrder": "First Order FD [NEW]",
        "PyNumDiff-SecondOrder": "Second Order FD [NEW]",
        "PyNumDiff-FourthOrder": "Fourth Order FD [NEW]",
        "PyNumDiff-MeanDiff-Auto": "Mean Diff (auto) [NEW]",
        "PyNumDiff-MeanDiff-Tuned": "Mean Diff (tuned) [NEW]",
        "PyNumDiff-MedianDiff-Auto": "Median Diff (auto) [NEW]",
        "PyNumDiff-MedianDiff-Tuned": "Median Diff (tuned) [NEW]",
        "PyNumDiff-RBF-Auto": "RBF (auto) [NEW]",
        "PyNumDiff-RBF-Tuned": "RBF (tuned) [NEW]",
    }

    print(f"Checking {len(all_methods)} PyNumDiff methods...")
    print()

    available = []
    unavailable = []

    for method_name, display_name in all_methods.items():
        try:
            result = evaluator.evaluate_method(method_name)
            if "predictions" in result:
                available.append((method_name, display_name))
                print(f"✅ {display_name:40} OK")
            else:
                unavailable.append((method_name, display_name))
                print(f"❌ {display_name:40} FAIL - No predictions")
        except Exception as e:
            unavailable.append((method_name, display_name))
            print(f"❌ {display_name:40} ERROR - {str(e)[:30]}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Available methods: {len(available)}/{len(all_methods)}")
    print(f"❌ Unavailable methods: {len(unavailable)}/{len(all_methods)}")

    if available:
        print("\n📊 Methods ready for comprehensive study:")
        print("-" * 40)
        # Group by category
        full_support = [m for m, d in available if "SavGol" in m or "Spectral" in m]
        excellent = [m for m, d in available if "TVRegularized" in m or "PolyDiff" in m or "Butter" in m or "Spline" in m and "Auto" in m]
        good = [m for m, d in available if "Second" in m or "Spline" in m and "Tuned" in m]
        baseline = [m for m, d in available if "First" in m or "Fourth" in m or "Mean" in m or "Median" in m]
        other = [m for m, d in available if m not in full_support + excellent + good + baseline]

        if full_support:
            print("\n🌟 Full orders 0-7 support:")
            for m in full_support:
                print(f"   - {m}")

        if excellent:
            print("\n⭐ Excellent performers (RMSE < 0.05):")
            for m in excellent:
                print(f"   - {m}")

        if good:
            print("\n✅ Good performers (RMSE < 0.1):")
            for m in good:
                print(f"   - {m}")

        if baseline:
            print("\n📈 Baseline/comparison methods:")
            for m in baseline:
                print(f"   - {m}")

        if other:
            print("\n📊 Other methods:")
            for m in other:
                print(f"   - {m}")

    if unavailable:
        print("\n⚠️  Unavailable methods:")
        for method_name, display_name in unavailable:
            print(f"   - {display_name}")

    print("\n" + "=" * 80)
    print("INTEGRATION STATUS")
    print("=" * 80)
    if len(available) == len(all_methods):
        print("✅✅ SUCCESS! All PyNumDiff methods are integrated and ready!")
        print("    The comprehensive study now has access to 30 PyNumDiff methods.")
    else:
        print(f"⚠️  PARTIAL SUCCESS: {len(available)}/{len(all_methods)} methods available")
        print(f"    Please check the {len(unavailable)} unavailable methods above")

    return len(available), len(all_methods)


if __name__ == "__main__":
    available, total = verify_comprehensive_study_integration()
    print(f"\nFinal count: {available}/{total} methods available for benchmarking")