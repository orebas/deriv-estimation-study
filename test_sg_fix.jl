#!/usr/bin/env julia
"""
Quick test of the fixed Savitzky-Golay implementation.

Tests on sin(x) where we know the exact derivatives:
- f(x) = sin(x)
- f'(x) = cos(x)
- f''(x) = -sin(x)
- f'''(x) = -cos(x)
"""

# Add project to path
using Pkg
Pkg.activate(".")

include("methods/julia/filtering/filters.jl")
using Printf

println("=" ^ 80)
println("TESTING FIXED SAVITZKY-GOLAY IMPLEMENTATION")
println("=" ^ 80)

# Test function: sin(x) on [0, 2π]
x = LinRange(0, 2π, 101)
y = sin.(x)

# Evaluation points (excluding endpoints to avoid boundary effects)
x_eval = x[11:91]  # Interior points

# Expected derivatives at π/2 (index ≈ 26)
test_idx = argmin(abs.(x_eval .- π / 2))
test_point = x_eval[test_idx]

println("\nTest point: x = $(test_point) (close to π/2 = $(π/2))")
println("\nExpected derivatives at x ≈ π/2:")
println("  f(x)    = sin(x)   ≈  1.0")
println("  f'(x)   = cos(x)   ≈  0.0")
println("  f''(x)  = -sin(x)  ≈ -1.0")
println("  f'''(x) = -cos(x)  ≈  0.0")

# Run the fixed SG method
orders = [0, 1, 2, 3]
result = evaluate_savitzky_golay(collect(x), y, collect(x_eval), orders)

println("\n" * "=" ^ 80)
println("RESULTS FROM FIXED SAVITZKY-GOLAY:")
println("=" ^ 80)

if result.success
	println("\n✓ Method executed successfully")

	for order in orders
		if haskey(result.predictions, order)
			pred = result.predictions[order]

			if length(pred) > 0 && !all(isnan.(pred))
				value_at_test = pred[test_idx]
				expected = [1.0, 0.0, -1.0, 0.0][order+1]
				error = abs(value_at_test - expected)

				@printf("\nOrder %d:\n", order)
				@printf("  Predicted:  %10.6f\n", value_at_test)
				@printf("  Expected:   %10.6f\n", expected)
				@printf("  Error:      %10.6f\n", error)

				if error < 0.1
					println("  Status:     ✓ PASS")
				else
					println("  Status:     ✗ FAIL (error too large)")
				end

				# Check if we're getting near-zero everywhere (old bug)
				non_nan = pred[.!isnan.(pred)]
				if length(non_nan) > 0
					pred_std = std(non_nan)
					pred_mean = mean(abs.(non_nan))
					@printf("  std(pred):  %10.6f\n", pred_std)
					@printf("  mean(|pred|): %10.6f\n", pred_mean)

					if pred_std < 1e-6 && order > 0
						println("  WARNING:    Predictions are nearly constant (possible bug!)")
					end
				end
			else
				println("\nOrder $order: No valid predictions (all NaN)")
			end
		else
			println("\nOrder $order: Failed")
			if haskey(result.failures, order)
				println("  Reason: $(result.failures[order])")
			end
		end
	end

	println("\n" * "=" ^ 80)
	println("SUMMARY")
	println("=" ^ 80)
	println("Timing: $(result.timing) seconds")

	if !isempty(result.failures)
		println("\nFailures:")
		for (order, msg) in result.failures
			println("  Order $order: $msg")
		end
	end

else
	println("✗ Method failed entirely")
	println("Failures: $(result.failures)")
end

println("\n" * "=" ^ 80)
println("TEST COMPLETE")
println("=" ^ 80)
