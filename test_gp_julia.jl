#!/usr/bin/env julia
"""
Quick test of GP-Julia-SE on noisy data.
"""

using Random
include("src/julia_methods.jl")

# Generate simple test data
Random.seed!(42)
x = collect(range(0, 10, length=50))
y_true = @. sin(x) + 0.5*cos(2*x)
noise_level = 0.01
y_noisy = y_true .+ noise_level .* randn(length(x))

println("Testing GP-Julia-SE with $(100*noise_level)% noise...")
println("Data points: $(length(x))")

# Test the analytic GP
try
    gp_func = fit_gp_se_analytic(x, y_noisy)

    # Evaluate at a few test points
    x_test = [2.0, 5.0, 8.0]

    println("\nEvaluating derivatives at test points:")
    for xi in x_test
        println("\nAt x = $xi:")
        for order in 0:3
            val = gp_func(xi, order)
            println("  Order $order: $val")
        end
    end

    println("\n✓ GP-Julia-SE succeeded!")

catch e
    println("\n✗ GP-Julia-SE failed:")
    println(e)
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end
