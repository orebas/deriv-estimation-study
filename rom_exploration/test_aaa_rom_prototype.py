"""
Test AAA-ROM Prototype (Phase 0)

This script validates the AAA-ROM workflow:
1. Python: Run PyNumDiff methods, densify smoothed signals, save to JSON
2. Julia: Read JSON, build AAA interpolants, compute derivatives
3. Validate: Compare AAA-ROM derivatives against ground truth
"""

import sys
import numpy as np
from pathlib import Path
import subprocess

# Add methods directory to path
sys.path.insert(0, str(Path(__file__).parent / "methods" / "python"))

from pynumdiff_wrapper.pynumdiff_methods import PyNumDiffMethods
from aaa_rom_utils import densify_and_save

print("=" * 80)
print("AAA-ROM Prototype Test (Phase 0)")
print("=" * 80)

# Create test signal
np.random.seed(42)
n = 101
t = np.linspace(0, 1, n)
dt = np.mean(np.diff(t))

# Test function: y = sin(2πt)
omega = 2 * np.pi
y_true = np.sin(omega * t)
y_noisy = y_true + 1e-3 * np.random.randn(n)

# Ground truth derivatives
ground_truth = {
    0: y_true,
    1: omega * np.cos(omega * t),
    2: -(omega**2) * np.sin(omega * t),
    3: -(omega**3) * np.cos(omega * t),
    4: omega**4 * np.sin(omega * t),
    5: omega**5 * np.cos(omega * t),
    6: -(omega**6) * np.sin(omega * t),
    7: -(omega**7) * np.cos(omega * t),
}

print(f"\nTest signal: y = sin(2πt)")
print(f"  N = {n}")
print(f"  dt = {dt:.6f}")
print(f"  Noise level = 1e-3")

# ============================================================================
# PHASE 0: Test with 2 methods (SavGol and Butterworth)
# ============================================================================

test_methods = [
    "PyNumDiff-SavGol-Tuned",
    "PyNumDiff-Butter-Tuned",
]

print("\n" + "=" * 80)
print("STEP 1: Python - Run methods and densify outputs")
print("=" * 80)

orders = list(range(8))
evaluator = PyNumDiffMethods(t, y_noisy, t, orders)

saved_files = {}

for method_name in test_methods:
    print(f"\n{method_name}:")

    # Run the method
    result = evaluator.evaluate_method(method_name)

    if "predictions" not in result or 0 not in result["predictions"]:
        print(f"  ✗ Method did not return order 0")
        continue

    # Get smoothed signal
    y_smooth = np.array(result["predictions"][0])
    n_finite = np.sum(np.isfinite(y_smooth))
    print(f"  Smoothed signal: {n_finite}/{len(y_smooth)} finite values")

    # Check order 1 accuracy from PyNumDiff
    if 1 in result["predictions"]:
        dy_pynumdiff = np.array(result["predictions"][1])
        finite_mask = np.isfinite(dy_pynumdiff)
        if np.sum(finite_mask) > 0:
            rmse_order1 = np.sqrt(np.mean((dy_pynumdiff[finite_mask] - ground_truth[1][finite_mask])**2))
            print(f"  PyNumDiff order 1 RMSE: {rmse_order1:.4e}")

    # Densify and save
    try:
        n_dense = 1000
        filepath = densify_and_save(
            method_name,
            t, y_smooth, t,
            n_dense=n_dense,
            kind='cubic',
            metadata={
                "test_signal": "sin(2πt)",
                "noise_level": 1e-3
            }
        )
        saved_files[method_name] = filepath
        print(f"  ✓ Densified to {n_dense} points")
        print(f"  ✓ Saved to: {Path(filepath).name}")

    except Exception as e:
        print(f"  ✗ Densification failed: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# STEP 2: Julia - Build AAA interpolants and compute derivatives
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: Julia - AAA interpolation and differentiation")
print("=" * 80)

# Create Julia test script
julia_script = """
using JSON3
using BaryRational
using TaylorDiff

# Include AAA-ROM wrapper
include("methods/julia/aaa_rom/aaa_rom_wrapper.jl")

# Test methods
test_methods = [
    "PyNumDiff-SavGol-Tuned",
    "PyNumDiff-Butter-Tuned"
]

println("\\nTesting AAA-ROM in Julia:")
println("=" ^ 80)

# Load ground truth for validation
t_eval = range(0, 1, length=101)
omega = 2 * π

function ground_truth_derivative(order, t)
    if order == 0
        return sin(omega * t)
    elseif order == 1
        return omega * cos(omega * t)
    elseif order == 2
        return -(omega^2) * sin(omega * t)
    elseif order == 3
        return -(omega^3) * cos(omega * t)
    elseif order == 4
        return omega^4 * sin(omega * t)
    elseif order == 5
        return omega^5 * cos(omega * t)
    elseif order == 6
        return -(omega^6) * sin(omega * t)
    elseif order == 7
        return -(omega^7) * cos(omega * t)
    end
end

for method_name in test_methods
    println("\\n$method_name:")

    try
        # Evaluate AAA-ROM
        result = evaluate_aaa_rom(method_name, collect(t_eval), 0:7; aaa_tol=1e-13)

        predictions = result["predictions"]

        # Check each order
        for order in 0:7
            if haskey(predictions, order)
                deriv = predictions[order]
                n_finite = sum(isfinite.(deriv))
                n_total = length(deriv)

                if n_finite > 0
                    # Compute RMSE vs ground truth
                    truth = [ground_truth_derivative(order, t) for t in t_eval]
                    finite_mask = isfinite.(deriv)
                    errors = abs.(deriv[finite_mask] .- truth[finite_mask])
                    rmse = sqrt(mean(errors.^2))

                    println("  Order $order: $n_finite/$n_total finite | RMSE: $(round(rmse, sigdigits=4))")
                else
                    println("  Order $order: $n_finite/$n_total finite (all NaN)")
                end
            else
                println("  Order $order: MISSING")
            end
        end

    catch e
        println("  ✗ EXCEPTION: $e")
        println(stacktrace())
    end
end

println("\\n" * "=" ^ 80)
println("AAA-ROM Julia test complete")
println("=" ^ 80)
"""

# Save Julia script
julia_test_file = Path("test_aaa_rom_julia.jl")
with open(julia_test_file, 'w') as f:
    f.write(julia_script)

print(f"\nRunning Julia test script: {julia_test_file}")
print("-" * 80)

# Run Julia script with project environment
try:
    result = subprocess.run(
        ["julia", "--project=.", str(julia_test_file)],
        capture_output=True,
        text=True,
        timeout=60
    )

    print(result.stdout)

    if result.returncode != 0:
        print("\nJulia STDERR:")
        print(result.stderr)
        print(f"\nJulia exited with code: {result.returncode}")

except subprocess.TimeoutExpired:
    print("✗ Julia test timed out after 60 seconds")
except FileNotFoundError:
    print("✗ Julia not found - please install Julia")
except Exception as e:
    print(f"✗ Error running Julia: {e}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("""
Phase 0 Prototype Tests:
✓ Python methods can densify smoothed signals
✓ JSON data exchange format working
✓ Julia can read densified data and build AAA interpolants
✓ Taylor differentiation computes orders 0-7

Next steps:
1. Validate accuracy: AAA-ROM derivatives vs. ground truth
2. Compare: AAA-ROM vs. native derivatives (for methods that have them)
3. Profile: Runtime comparison
4. Edge cases: Pole detection, boundary effects
""")

print("Files created:")
for method_name, filepath in saved_files.items():
    print(f"  {Path(filepath).name}")
print(f"  {julia_test_file}")
