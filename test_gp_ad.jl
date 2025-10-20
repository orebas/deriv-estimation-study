#!/usr/bin/env julia
"""
Quick test of GP-Julia-AD on noisy data.
Tests the TaylorDiff-based automatic differentiation approach.
"""

using Random
include("src/julia_methods.jl")

# Generate simple test data
Random.seed!(42)
x = collect(range(0, 10, length=50))
y_true = @. sin(x) + 0.5*cos(2*x)
noise_level = 0.01
y_noisy = y_true .+ noise_level .* randn(length(x))

println("Testing GP-Julia-AD with $(100*noise_level)% noise...")
println("Data points: $(length(x))")

# Test the AD-based GP
try
    gp_func = fit_gp_ad(x, y_noisy)

    # Evaluate at a few test points
    x_test = [2.0, 5.0, 8.0]

    println("\nEvaluating derivatives at test points:")
    for xi in x_test
        println("\nAt x = $xi:")
        for order in 0:3
            val = nth_deriv_taylor(gp_func, order, xi)
            println("  Order $order: $val")
        end
    end

    println("\n✓ GP-Julia-AD succeeded!")

catch e
    println("\n✗ GP-Julia-AD failed:")
    println(e)
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end
