#!/usr/bin/env julia
"""
Test the fixes to TVRegDiff and Fourier-Interp methods.

Verifies:
1. TVRegDiff now computes derivatives for orders 4-7 (not just 0-3)
2. Fourier-Interp uses stable FFT-based implementation
"""

using Random
using Printf
include("src/julia_methods.jl")

println("="^70)
println("Testing Fixed Methods: TVRegDiff and Fourier-Interp")
println("="^70)

# Generate test data with known derivatives
Random.seed!(42)
x = collect(range(0, 10, length=100))
y_true = @. sin(x) + 0.5*cos(2*x)
noise_level = 0.01
y_noisy = y_true .+ noise_level .* randn(length(x))

println("Data points: $(length(x))")
println("Noise level: $(100*noise_level)%")
println()

# Test points (avoid endpoints)
x_test = collect(range(2.0, 8.0, length=10))

# Ground truth derivatives at x=5.0
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

# Test 1: TVRegDiff for orders 0-7
println("TEST 1: TVRegDiff (Rick Chartrand's method)")
println("-"^70)
println("Testing orders 0-7 (previously failed on 4-7)...")
println()

try
    # Use the same approach as evaluate_julia_method
    orders = collect(0:7)
    alpha = 0.05

    # Fit TVRegDiff (precomputes up to max order)
    fitted_deriv_funcs = fit_tvregdiff(x, y_noisy, orders, alpha)

    println("  Order  | Value at x=5.0 | Ground Truth | Error")
    println("  " * "-"^58)

    all_success = true
    for order in 0:7
        # Evaluate at x=5.0
        deriv_func = fitted_deriv_funcs[order]
        predicted = deriv_func(5.0)
        truth = ground_truth[order]
        error = abs(predicted - truth)

        if isnan(predicted) || isinf(predicted)
            @printf("  %-6d | %14s | %12.6f | FAILED (NaN/Inf)\n",
                    order, "NaN/Inf", truth)
            all_success = false
        else
            @printf("  %-6d | %14.6f | %12.6f | %.2e\n",
                    order, predicted, truth, error)
        end
    end

    println()
    if all_success
        println("  ✓ TVRegDiff: All orders (0-7) computed successfully!")
    else
        println("  ✗ TVRegDiff: Some orders failed!")
    end
    println()

catch e
    println("  ✗ TVRegDiff failed with error:")
    println("    $e")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
    println()
end

# Test 2: Fourier-Interp FFT-based implementation
println("TEST 2: Fourier-Interp (FFT-based spectral method)")
println("-"^70)
println("Testing numerical stability of FFT implementation...")
println()

try
    # Fit using new FFT-based method
    fourier_fft = fit_fourier(x, y_noisy)

    println("  Order  | Value at x=5.0 | Ground Truth | Error")
    println("  " * "-"^58)

    all_success = true
    max_error = 0.0

    for order in 0:7
        # Evaluate at x=5.0 using FFT-based derivative
        predicted = fourier_fft_deriv(fourier_fft, 5.0, order)
        truth = ground_truth[order]
        error = abs(predicted - truth)
        max_error = max(max_error, error)

        if isnan(predicted) || isinf(predicted)
            @printf("  %-6d | %14s | %12.6f | FAILED (NaN/Inf)\n",
                    order, "NaN/Inf", truth)
            all_success = false
        else
            @printf("  %-6d | %14.6f | %12.6f | %.2e\n",
                    order, predicted, truth, error)
        end
    end

    println()
    if all_success
        println("  ✓ Fourier-Interp: All orders computed without NaN/Inf!")
        println("  ✓ Maximum error: $(max_error)")
    else
        println("  ✗ Fourier-Interp: Some orders failed!")
    end
    println()

catch e
    println("  ✗ Fourier-Interp failed with error:")
    println("    $e")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
    println()
end

println("="^70)
println("Summary:")
println("  - TVRegDiff should now support orders 4-7 (was hard-coded to 3)")
println("  - Fourier-Interp should use stable FFT (not ill-conditioned Vandermonde)")
println("="^70)
