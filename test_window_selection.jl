"""
Test window selection with fixed noise estimator.
"""

using Random
using Statistics

function estimate_noise_diff2(y)
    n = length(y)
    d = y[3:end] - 2.0 * y[2:end-1] + y[1:end-2]
    return median(abs.(d .- median(d))) / 0.6745 / sqrt(6.0)
end

function test_function(x)
    return sin.(2π * x) .+ 0.5 * sin.(4π * x)
end

Random.seed!(12345)
n = 101
x = collect(range(0, 1, length=n))
y_true = test_function(x)
dx = mean(diff(x))

noise_levels = [1e-8, 1e-6, 1e-4, 1e-3]

# Calibration constants
c_pr = Dict(
    (7, 0) => 1.0, (7, 1) => 1.1, (7, 2) => 1.2, (7, 3) => 1.3,
    (9, 4) => 1.4, (9, 5) => 1.5,
    (11, 6) => 1.6, (11, 7) => 1.7,
)

println("Window Selection with FIXED Noise Estimator:")
println("=" ^ 80)

for noise_std in noise_levels
    y_noisy = y_true .+ noise_std * randn(n)

    σ_hat = estimate_noise_diff2(y_noisy)

    # Roughness estimate
    d4 = y_noisy
    for _ in 1:4
        d4 = diff(d4)
    end
    ρ_hat = sqrt(mean(d4 .^ 2)) / (dx^4 + 1e-24)

    ratio = max(σ_hat, 1e-24)^2 / max(ρ_hat, 1e-24)^2

    println("\nNoise: σ = $noise_std")
    println("  σ̂ = $(round(σ_hat, sigdigits=4)), ρ̂ = $(round(ρ_hat, sigdigits=4))")
    println("  σ̂²/ρ̂² = $(round(ratio, sigdigits=4))")
    println()
    println("  Window sizes (adaptive vs fixed w=15):")

    for order in 0:7
        p = if order <= 3; 7 elseif order <= 5; 9 else 11 end
        c = get(c_pr, (p, order), 1.0 + 0.1 * order)

        h_star = c * (ratio^(1.0 / (2 * p + 3)))
        w_ideal = Int(2 * floor(h_star / max(dx, 1e-24)) + 1)

        # Apply constraints
        max_window = 33  # n/3
        w = max(w_ideal, p + 3)
        w = min(w, max_window)
        w = min(w, n)

        if iseven(w)
            w -= 1
        end

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

        marker = if w < 15
            "✓ smaller"
        elseif w == 15
            "= same"
        else
            "  larger"
        end

        println("    Order $order: w=$w (poly=$p) vs Fixed w=15 $marker")
    end
end
