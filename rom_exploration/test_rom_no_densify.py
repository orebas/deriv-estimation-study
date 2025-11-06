"""
Test ROM WITHOUT densification

Hypothesis: Cubic spline densification introduces artifacts that blow up in higher derivatives.
Solution: Fit the original smoothed points directly with ApproxFun.
"""

import sys
import numpy as np
from pathlib import Path
import subprocess

# Add methods directory to path
sys.path.insert(0, str(Path(__file__).parent / "methods" / "python"))

from pynumdiff_wrapper.pynumdiff_methods import PyNumDiffMethods

print("=" * 80)
print("ROM WITHOUT Densification Test")
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

print(f"\nTest signal: y = sin(2πt)")
print(f"  N = {n}")
print(f"  Noise level = 1e-3")

# Run PyNumDiff
evaluator = PyNumDiffMethods(t, y_noisy, t, list(range(8)))
result = evaluator.evaluate_method("PyNumDiff-SavGol-Tuned")

y_smooth = np.array(result["predictions"][0])

# Save original (non-densified) data
import json
output_dir = Path("build/data/rom_input")
output_dir.mkdir(parents=True, exist_ok=True)

data = {
    "method_name": "PyNumDiff-SavGol-Tuned-NoDensify",
    "t_dense": t.tolist(),  # Original points, not densified
    "y_dense": y_smooth.tolist(),
    "t_eval": t.tolist(),
    "metadata": {
        "n_dense": len(t),
        "densification": "none",
        "test_signal": "sin(2πt)",
        "noise_level": 1e-3
    }
}

filepath = output_dir / "PyNumDiff-SavGol-Tuned-NoDensify_dense.json"
with open(filepath, 'w') as f:
    json.dump(data, f, indent=2)

print(f"\n✓ Saved {len(t)} original points (no densification)")
print(f"✓ File: {filepath.name}")

# Julia test
julia_script = """
using ApproxFun
using JSON3
using Statistics

include("methods/julia/rom/approxfun_rom_wrapper.jl")

println("\\nTesting ROM WITHOUT densification:")
println("=" ^ 80)

# Load ground truth
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

# Test with different polynomial degrees
for max_degree in [10, 20, 30]
    println("\\nDegree $max_degree:")

    try
        result = evaluate_rom("PyNumDiff-SavGol-Tuned-NoDensify", collect(t_eval), 0:7; max_degree=max_degree)

        predictions = result["predictions"]

        for order in 0:7
            if haskey(predictions, order)
                deriv = predictions[order]
                truth = [ground_truth_derivative(order, t) for t in t_eval]
                finite_mask = isfinite.(deriv)

                if sum(finite_mask) > 0
                    errors = abs.(deriv[finite_mask] .- truth[finite_mask])
                    rmse = sqrt(mean(errors.^2))
                    println("  Order $order: RMSE = $(round(rmse, sigdigits=4))")
                else
                    println("  Order $order: All NaN")
                end
            end
        end

    catch e
        println("  ERROR: $e")
    end
end

println("\\n" * "=" ^ 80)
"""

julia_test_file = Path("test_rom_no_densify.jl")
with open(julia_test_file, 'w') as f:
    f.write(julia_script)

print(f"\nRunning Julia test: {julia_test_file}")
print("-" * 80)

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

except Exception as e:
    print(f"Error running Julia: {e}")

print("\n" * "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("""
If this works better, then densification is the problem!

We should:
1. Skip densification entirely
2. Fit original smoothed points directly
3. Use controlled polynomial degree (10-30)
""")
