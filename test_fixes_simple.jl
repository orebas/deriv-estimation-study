#!/usr/bin/env julia
"""
Simple test to verify TVRegDiff now supports orders 4-7.
Uses evaluate_julia_method directly (the actual production code path).
"""

using Random
using Printf
include("src/julia_methods.jl")

println("="^70)
println("Testing TVRegDiff Fix: Order Support 0-7")
println("="^70)

# Generate test data
Random.seed!(42)
x = collect(range(0, 10, length=100))
y_true = @. sin(x) + 0.5*cos(2*x)
noise_level = 0.01
y_noisy = y_true .+ noise_level .* randn(length(x))

# Test points
x_test = [5.0]  # Just test at one point for simplicity

# Ground truth at x=5.0
ground_truth = Dict(
    0 => sin(5.0) + 0.5*cos(10.0),
    1 => cos(5.0) - sin(10.0),
    2 => -sin(5.0) - 2*cos(10.0),
    3 => -cos(5.0) + 4*sin(10.0),
    4 => sin(5.0) + 8*cos(10.0),
    5 => cos(5.0) - 16*sin(10.0),
    6 => -sin(5.0) - 32*cos(10.0),
    7 => -cos(5.0) + 64*sin(10.0)
)

println("Data: $(length(x)) points, $(100*noise_level)% noise")
println("Testing derivative orders 0-7 at x=5.0")
println()

# Test using the actual evaluate_julia_method function
println("Running TVRegDiff-Julia via evaluate_julia_method...")
result = evaluate_julia_method("TVRegDiff-Julia", x, y_noisy, x_test, collect(0:7))

println()
println("Results:")
println("-"^70)
println("Order  | Predicted    | Truth        | Error       | Status")
println("-"^70)

all_success = true
for order in 0:7
    if haskey(result.predictions, order) && !isempty(result.predictions[order])
        pred = result.predictions[order][1]  # First (only) test point
        truth = ground_truth[order]
        error = abs(pred - truth)

        if isnan(pred) || isinf(pred)
            status = "FAILED (NaN/Inf)"
            all_success = false
        else
            status = "✓"
        end

        @printf("%-6d | %12.6f | %12.6f | %.2e | %s\n",
                order, pred, truth, error, status)
    else
        println("$order      | MISSING")
        all_success = false
    end
end

println("-"^70)
println()

if all_success
    println("✓ SUCCESS: TVRegDiff now supports all orders 0-7!")
    println("  (Previously hard-coded to max_order_precompute = 3)")
else
    println("✗ FAILURE: Some orders failed")
    if !isempty(result.failures)
        println("\nFailure details:")
        for (order, msg) in result.failures
            println("  Order $order: $msg")
        end
    end
end

println("="^70)
