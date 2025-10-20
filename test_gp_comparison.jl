#!/usr/bin/env julia
"""
Compare GP-Julia-SE vs GP-Julia-AD on noisy data.
"""

using Random
using Printf
include("src/julia_methods.jl")

# Generate test data (Lotka-Volterra observable simulation)
Random.seed!(42)
x = collect(range(0, 10, length=100))
y_true = @. sin(x) + 0.5*cos(2*x)
noise_level = 0.01
y_noisy = y_true .+ noise_level .* randn(length(x))

# Ground truth 3rd derivative
y_deriv3_true = @. -cos(x) + 4*sin(2*x)

println("="^70)
println("GP Method Comparison: GP-Julia-SE vs GP-Julia-AD")
println("="^70)
println("Data points: $(length(x))")
println("Noise level: $(100*noise_level)%")
println()

# Test points (avoid endpoints)
x_test = collect(range(2.0, 8.0, length=20))

# Test GP-Julia-SE
println("Testing GP-Julia-SE (Hermite polynomial approach)...")
try
    gp_se = fit_gp_se_analytic(x, y_noisy)

    # Compute 3rd derivatives
    pred_se = [gp_se(xi, 3) for xi in x_test]
    true_vals = [(-cos(xi) + 4*sin(2*xi)) for xi in x_test]

    # Compute RMSE
    rmse_se = sqrt(sum((pred_se .- true_vals).^2) / length(pred_se))

    # Check for NaNs
    num_nans_se = sum(isnan.(pred_se))

    println("  ✓ GP-Julia-SE completed")
    println("  RMSE (3rd deriv): $(round(rmse_se, digits=4))")
    println("  NaN values: $num_nans_se / $(length(pred_se))")
    println()
catch e
    println("  ✗ GP-Julia-SE failed: $e")
    println()
end

# Test GP-Julia-AD
println("Testing GP-Julia-AD (TaylorDiff AD approach)...")
try
    gp_ad = fit_gp_ad(x, y_noisy)

    # Compute 3rd derivatives using TaylorDiff
    pred_ad = [nth_deriv_taylor(gp_ad, 3, xi) for xi in x_test]
    true_vals = [(-cos(xi) + 4*sin(2*xi)) for xi in x_test]

    # Compute RMSE
    rmse_ad = sqrt(sum((pred_ad .- true_vals).^2) / length(pred_ad))

    # Check for NaNs
    num_nans_ad = sum(isnan.(pred_ad))

    println("  ✓ GP-Julia-AD completed")
    println("  RMSE (3rd deriv): $(round(rmse_ad, digits=4))")
    println("  NaN values: $num_nans_ad / $(length(pred_ad))")
    println()
catch e
    println("  ✗ GP-Julia-AD failed: $e")
    println()
end

println("="^70)
