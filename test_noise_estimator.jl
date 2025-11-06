"""
Compare OLD (1st-order) vs NEW (2nd-order) noise estimators.
"""

using Random
using Statistics

# Implement both estimators inline
function estimate_noise_diff1(y)
    dy = diff(y)
    return median(abs.(dy .- median(dy))) / (0.6745 * sqrt(2.0))
end

function estimate_noise_diff2(y)
    n = length(y)
    if n < 3
        error("Need at least 3 points")
    end
    # Second-order difference operator
    d = y[3:end] - 2.0 * y[2:end-1] + y[1:end-2]
    # MAD estimator: Var(d) = 6σ²
    return median(abs.(d .- median(d))) / 0.6745 / sqrt(6.0)
end

# Test function
function test_function(x)
    return sin.(2π * x) .+ 0.5 * sin.(4π * x)
end

# Generate test data
Random.seed!(12345)
n = 101
x = collect(range(0, 1, length=n))
y_true = test_function(x)

noise_levels = [1e-8, 1e-6, 1e-4, 1e-3]

println("Comparing Noise Estimators:")
println("=" ^ 80)
println()

for noise_std in noise_levels
    y_noisy = y_true .+ noise_std * randn(n)

    σ_old = estimate_noise_diff1(y_noisy)
    σ_new = estimate_noise_diff2(y_noisy)

    error_old = abs(σ_old - noise_std)
    error_new = abs(σ_new - noise_std)

    println("Actual noise: σ = $noise_std")
    println("  OLD (1st-order): σ̂ = $(round(σ_old, sigdigits=4))  (error: $(round(error_old, sigdigits=3)))")
    println("  NEW (2nd-order): σ̂ = $(round(σ_new, sigdigits=4))  (error: $(round(error_new, sigdigits=3)))")

    if error_new < error_old * 0.5
        println("  ✓ NEW is much better ($(round(error_old/error_new, digits=1))× more accurate)")
    elseif error_new < error_old
        println("  ✓ NEW is better")
    else
        println("  ✗ OLD is better")
    end
    println()
end
