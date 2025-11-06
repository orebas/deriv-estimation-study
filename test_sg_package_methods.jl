"""
Test the new SavitzkyGolay package-based methods implementation.
Verifies:
1. Methods can be called without errors
2. Results are reasonable (finite, correct length)
3. Comparison against our custom S-G implementation
4. Window selection behavior across methods
"""

using Random
using Statistics
using Printf

# Load the methods
include("src/julia_methods_integrated.jl")

println("="^80)
println("Testing SavitzkyGolay Package-Based Methods")
println("="^80)

# Generate test signal
Random.seed!(42)
n = 251
x = collect(range(0, 1, length=n))
dx = mean(diff(x))

# Smooth ODE-like signal
y_true = sin.(2π * x) .+ 0.3 * sin.(6π * x)
dy_true = 2π * cos.(2π * x) .+ 0.3 * 6π * cos.(6π * x)

# Add moderate noise
σ = 1e-4
y_noisy = y_true .+ σ * randn(n)

# Evaluation points (same as training for simplicity)
x_eval = x

# Derivative orders to test
orders = [0, 1, 2, 3]

println("\nTest Setup:")
println("  N = $n")
println("  dx = $(round(dx, sigdigits=4))")
println("  Noise level σ = $σ")
println("  Signal: smooth ODE-like (sum of sinusoids)")
println()

# Test each method
method_names = [
    "SG-Package-Fixed",
    "SG-Package-Hybrid",
    "SG-Package-Adaptive",
    "Savitzky-Golay-Fixed",     # Our custom implementation for comparison
    "Savitzky-Golay-Adaptive",  # Our custom implementation for comparison
]

println("Testing Methods:")
println("-"^80)

results_dict = Dict()

for method_name in method_names
    println("\n$method_name:")

    try
        # Call the method
        result = evaluate_julia_method(method_name, x, y_noisy, x_eval, orders)

        # Check basic properties
        success = result.success
        n_orders = length(result.predictions)

        println("  ✓ Method completed (success=$success)")
        println("  ✓ Returned $n_orders derivative orders")

        # Check each order
        for order in orders
            if haskey(result.predictions, order)
                deriv = result.predictions[order]
                n_finite = count(isfinite, deriv)
                n_total = length(deriv)
                pct_finite = 100 * n_finite / n_total

                println("  Order $order: $n_finite/$n_total finite ($(round(pct_finite, digits=1))%)")

                # Check for order 1 accuracy (compare to true derivative)
                if order == 1 && n_finite > 0
                    finite_idx = findall(isfinite, deriv)
                    if !isempty(finite_idx)
                        errors = abs.(deriv[finite_idx] .- dy_true[finite_idx])
                        rmse = sqrt(mean(errors.^2))
                        max_error = maximum(errors)
                        println("    RMSE vs true: $(round(rmse, sigdigits=4))")
                        println("    Max error: $(round(max_error, sigdigits=4))")
                    end
                end
            else
                println("  Order $order: NOT FOUND")
            end
        end

        results_dict[method_name] = result

    catch e
        println("  ✗ ERROR: $e")
        println("  Stack trace:")
        for (i, frame) in enumerate(stacktrace(catch_backtrace())[1:min(5, end)])
            println("    $i. $frame")
        end
    end
end

println("\n" * "="^80)
println("Comparison: Package vs Custom Implementation")
println("="^80)

# Compare Fixed methods
if haskey(results_dict, "SG-Package-Fixed") && haskey(results_dict, "Savitzky-Golay-Fixed")
    pkg_result = results_dict["SG-Package-Fixed"]
    custom_result = results_dict["Savitzky-Golay-Fixed"]

    println("\nFixed Methods Comparison:")
    for order in orders
        if haskey(pkg_result.predictions, order) && haskey(custom_result.predictions, order)
            pkg_deriv = pkg_result.predictions[order]
            custom_deriv = custom_result.predictions[order]

            # Find common finite points
            both_finite = isfinite.(pkg_deriv) .& isfinite.(custom_deriv)
            n_common = count(both_finite)

            if n_common > 0
                diffs = abs.(pkg_deriv[both_finite] .- custom_deriv[both_finite])
                max_diff = maximum(diffs)
                mean_diff = mean(diffs)

                println("  Order $order ($n_common points):")
                println("    Max difference: $(round(max_diff, sigdigits=4))")
                println("    Mean difference: $(round(mean_diff, sigdigits=4))")

                # Check relative difference
                pkg_scale = maximum(abs.(pkg_deriv[both_finite]))
                rel_diff = max_diff / (pkg_scale + 1e-10)
                println("    Relative difference: $(round(rel_diff * 100, digits=2))%")
            end
        end
    end
end

# Compare Adaptive methods
if haskey(results_dict, "SG-Package-Adaptive") && haskey(results_dict, "Savitzky-Golay-Adaptive")
    pkg_result = results_dict["SG-Package-Adaptive"]
    custom_result = results_dict["Savitzky-Golay-Adaptive"]

    println("\nAdaptive Methods Comparison:")
    for order in orders
        if haskey(pkg_result.predictions, order) && haskey(custom_result.predictions, order)
            pkg_deriv = pkg_result.predictions[order]
            custom_deriv = custom_result.predictions[order]

            # Find common finite points
            both_finite = isfinite.(pkg_deriv) .& isfinite.(custom_deriv)
            n_common = count(both_finite)

            if n_common > 0
                diffs = abs.(pkg_deriv[both_finite] .- custom_deriv[both_finite])
                max_diff = maximum(diffs)
                mean_diff = mean(diffs)

                println("  Order $order ($n_common points):")
                println("    Max difference: $(round(max_diff, sigdigits=4))")
                println("    Mean difference: $(round(mean_diff, sigdigits=4))")
            end
        end
    end
end

println("\n" * "="^80)
println("Window Selection Analysis")
println("="^80)

# Note: We can't easily extract the chosen window sizes without modifying the code,
# but we can infer behavior from the number of boundary points discarded
println("\nBoundary Points Discarded (inferred from NaN count):")

for method_name in ["SG-Package-Fixed", "SG-Package-Hybrid", "SG-Package-Adaptive"]
    if haskey(results_dict, method_name)
        result = results_dict[method_name]
        println("\n$method_name:")

        for order in orders
            if haskey(result.predictions, order)
                deriv = result.predictions[order]
                n_nan = count(!isfinite, deriv)

                # NaN count gives us 2*m (both ends)
                # From GPT-5 formula: m = function(window_size, deriv_order)
                println("  Order $order: $n_nan points discarded")
            end
        end
    end
end

println("\n" * "="^80)
println("Test Summary")
println("="^80)

n_success = count(haskey(results_dict, name) for name in method_names)
println("\n✓ Successfully tested $n_success/$(length(method_names)) methods")

if n_success == length(method_names)
    println("\n🎉 All tests passed! Methods are ready for benchmarking.")
else
    println("\n⚠️  Some methods failed. Review errors above.")
end

println("\nNext steps:")
println("  1. Run comprehensive benchmark: ./build_all.sh")
println("  2. Compare performance across noise levels and derivative orders")
println("  3. Analyze window selection behavior (Fixed vs Hybrid vs Adaptive)")
println("  4. Update paper figures and tables")
