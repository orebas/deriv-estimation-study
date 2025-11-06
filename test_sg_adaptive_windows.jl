"""
Diagnose why Savitzky-Golay-Adaptive performs worse than Fixed.
Check what window sizes are being chosen.
"""

using Random
using Statistics

# Load noise estimator
include("src/hyperparameter_selection.jl")
using .HyperparameterSelection

# Simple test function with known derivatives
function test_function(x)
    # f(x) = sin(2πx) + 0.5*sin(4πx)
    return sin.(2π * x) .+ 0.5 * sin.(4π * x)
end

# Generate test data
Random.seed!(12345)
n = 101
x = collect(range(0, 1, length=n))
y_true = test_function(x)

# Add noise levels matching the study
noise_levels = [1e-8, 1e-6, 1e-4, 1e-3]

println("Testing Savitzky-Golay window selection:")
println("=" ^ 70)

for noise_std in noise_levels
    y_noisy = y_true .+ noise_std * randn(n)

    println("\nNoise level: $noise_std (actual)")
    println("-" ^ 70)

    # Test adaptive method and extract window sizes
    # Compare OLD (1st-order diff) vs NEW (2nd-order diff) estimators

    # OLD broken estimator (1st-order differences)
    dy = diff(y_noisy)
    σ_hat_old = median(abs.(dy .- median(dy))) / (0.6745 * sqrt(2.0))

    # NEW proper estimator (2nd-order differences)
    σ_hat = HyperparameterSelection.estimate_noise_diff2(y_noisy)

    println("  OLD (1st-order diff) σ̂ = $(round(σ_hat_old, sigdigits=4)) [BROKEN]")
    println("  NEW (2nd-order diff) σ̂ = $(round(σ_hat, sigdigits=4)) [CORRECT]")

    # Roughness estimate
    dx = mean(diff(x))
    d4 = y_noisy
    for _ in 1:4
        d4 = diff(d4)
    end
    ρ_hat = sqrt(mean(d4 .^ 2)) / (dx^4 + 1e-24)

    println("  Estimated roughness ρ̂ = $(round(ρ_hat, sigdigits=4))")
    println("  σ̂²/ρ̂² ratio = $(round((σ_hat^2)/(ρ_hat^2), sigdigits=4))")

    # Calibration constants
    c_pr = Dict(
        (7, 0) => 1.0, (7, 1) => 1.1, (7, 2) => 1.2, (7, 3) => 1.3,
        (9, 0) => 1.0, (9, 1) => 1.1, (9, 2) => 1.2, (9, 3) => 1.3, (9, 4) => 1.4, (9, 5) => 1.5,
        (11, 0) => 1.0, (11, 1) => 1.1, (11, 2) => 1.2, (11, 3) => 1.3, (11, 4) => 1.4,
        (11, 5) => 1.5, (11, 6) => 1.6, (11, 7) => 1.7,
    )

    max_window = max(5, div(n, 3))
    if iseven(max_window)
        max_window -= 1
    end

    println("\n  Adaptive window choices (vs fixed w=15):")
    for order in 0:7
        # Determine polynomial order
        p = if order <= 3
            7
        elseif order <= 5
            9
        else
            11
        end

        c = get(c_pr, (p, order), 1.0 + 0.1 * order)

        # Compute optimal window
        ratio = max(σ_hat, 1e-24)^2 / max(ρ_hat, 1e-24)^2
        h_star = c * (ratio^(1.0 / (2 * p + 3)))

        # Convert to window length
        w_ideal = Int(2 * floor(h_star / max(dx, 1e-24)) + 1)

        # Apply constraints
        w = max(w_ideal, p + 3)
        w = min(w, max_window)
        w = min(w, n)

        # Ensure odd
        if iseven(w)
            w -= 1
        end

        # Final check
        if w <= p
            w = p + 2
            if iseven(w)
                w += 1
            end
            w = min(w, n)
            if iseven(w)
                w -= 1
            end
        end

        println("    Order $order: w=$w (poly=$p) vs Fixed w=15 (poly=7)")
    end
end
