"""
Test different window sizes for N=251 to find empirical optimum.
"""

using Random
using Statistics
using LinearAlgebra

# Simple S-G implementation
function savitzky_golay_coeffs(window::Int, polyorder::Int, deriv_order::Int = 0)
    half_window = window ÷ 2
    pos = collect((-half_window):half_window)
    A = zeros(window, polyorder + 1)
    for i = 1:window, j = 0:polyorder
        A[i, j+1] = pos[i]^j
    end

    e_k = zeros(polyorder + 1)
    e_k[deriv_order+1] = 1.0

    c = (A' * A) \ e_k
    filter_weights = A * c * factorial(deriv_order)
    return filter_weights
end

function apply_sg(y, window, polyorder, deriv_order, dx)
    n = length(y)
    half_window = window ÷ 2
    weights = savitzky_golay_coeffs(window, polyorder, deriv_order)

    result = fill(NaN, n)
    for i = 1:n
        if i <= half_window || i > n - half_window
            continue
        end
        window_data = y[(i-half_window):(i+half_window)]
        result[i] = dot(weights, window_data) / (dx^deriv_order)
    end

    return result
end

# Test function with known derivatives
function test_sin(x)
    return sin.(2π * x) .+ 0.5 * sin.(4π * x)
end

function test_sin_deriv(x, order)
    if order == 0
        return test_sin(x)
    elseif order == 1
        return 2π * cos.(2π * x) .+ π * cos.(4π * x)
    elseif order == 2
        return -4π^2 * sin.(2π * x) .- 2π^2 * sin.(4π * x)
    else
        # Use finite differences for higher orders
        dx = x[2] - x[1]
        y = test_sin_deriv(x, order - 1)
        return diff(y) / dx
    end
end

Random.seed!(42)
n = 251
x = collect(range(0, 1, length=n))
dx = mean(diff(x))
y_true = test_sin(x)

# Test different noise levels
noise_levels = [1e-6, 1e-4, 1e-3]

# Test window sizes
window_sizes = [11, 15, 21, 25, 31, 35, 39, 45, 51]

println("Optimal Window Search for N=251")
println("=" ^ 80)

for noise_std in noise_levels
    println("\nNoise level: σ = $noise_std")
    println("-" ^ 80)

    # Test on orders 0, 2, 4
    for deriv_order in [0, 2, 4]
        y_noisy = y_true .+ noise_std * randn(n)
        y_true_deriv = test_sin_deriv(x, deriv_order)

        # Trim to interior where we can evaluate
        interior_start = 26  # Skip first 10%
        interior_end = n - 25

        best_rmse = Inf
        best_window = 0

        results = []

        for w in window_sizes
            if w > n
                continue
            end

            polyorder = deriv_order <= 3 ? 7 : (deriv_order <= 5 ? 9 : 11)

            if w <= polyorder
                continue
            end

            y_est = apply_sg(y_noisy, w, polyorder, deriv_order, dx)

            # Compute RMSE on interior
            valid = interior_start:interior_end
            mask = .!isnan.(y_est[valid])

            if sum(mask) == 0
                continue
            end

            rmse = sqrt(mean((y_est[valid][mask] .- y_true_deriv[valid][mask]).^2))

            push!(results, (w, rmse))

            if rmse < best_rmse
                best_rmse = rmse
                best_window = w
            end
        end

        println("\n  Order $deriv_order:")
        for (w, rmse) in results
            marker = w == best_window ? " ← BEST" : ""
            println("    w=$w: RMSE=$(round(rmse, sigdigits=4))$marker")
        end
    end
end

println("\n" * "=" ^ 80)
println("Summary: For N=251, optimal window appears to be ~15-21% of N")
println("  N=251 → w ≈ 38-53 (but results above show actual optimum)")
