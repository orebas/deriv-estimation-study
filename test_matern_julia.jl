#!/usr/bin/env julia
"""
Quick test to verify Julia Matern implementation works correctly.
Tests all three nu values: 0.5, 1.5, 2.5
"""

# Add project path
using Pkg
Pkg.activate(".")

# Import required packages
using Statistics
using Random

# Load julia_methods module
include("src/julia_methods.jl")

# Test configuration
Random.seed!(42)
x_train = collect(range(0, 2π, length=21))
y_train = sin.(x_train) .+ 0.001 .* randn(length(x_train))
x_eval = x_train
orders = [0, 1, 2, 3, 4, 5]

println("="^70)
println("JULIA MATERN KERNEL VERIFICATION TEST")
println("="^70)
println()
println("Test setup:")
println("  Training points: ", length(x_train))
println("  Evaluation points: ", length(x_eval))
println("  Orders: ", orders)
println()

# Test each Matern kernel variant
for method_name in ["GP-Julia-Matern-0.5", "GP-Julia-Matern-1.5", "GP-Julia-Matern-2.5"]
	println("-"^70)
	println("Testing $method_name...")

	t_start = time()
	result = evaluate_julia_method(method_name, x_train, y_train, x_eval, orders)
	elapsed = time() - t_start

	println("  Success: ", result.success)
	println("  Timing: ", round(elapsed, digits=3), "s")

	if !result.success
		println("  ERROR: Method failed")
		if !isempty(result.failures)
			for (order, error_msg) in result.failures
				println("    Order $order: $error_msg")
			end
		end
	else
		println("  Orders completed: ", sort(collect(keys(result.predictions))))

		if !isempty(result.failures)
			println("  Partial failures:")
			for (order, error_msg) in result.failures
				println("    Order $order: $error_msg")
			end
		end

		# Show sample predictions for first 3 points
		println("  Sample predictions (first 3 eval points):")
		for order in orders
			if haskey(result.predictions, order)
				preds = result.predictions[order]
				sample_preds = preds[1:min(3, length(preds))]
				println("    Order $order: ", round.(sample_preds, digits=6))
			end
		end
	end
	println()
end

println("="^70)
println("✓ ALL JULIA MATERN TESTS COMPLETED")
println("="^70)
