"""
Compare SavitzkyGolay.jl package vs our custom implementation.
"""

using Random
using Statistics
using LinearAlgebra
using SavitzkyGolay

# Our implementation
function our_sg_coeffs(window::Int, polyorder::Int, deriv_order::Int = 0)
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

# Test signal
Random.seed!(42)
n = 101
x = collect(range(0, 1, length=n))
dx = mean(diff(x))
y_true = sin.(2π * x) .+ 0.5 * sin.(4π * x)
y_noisy = y_true .+ 1e-4 * randn(n)

println("Comparing SavitzkyGolay.jl package vs our implementation")
println("=" ^ 80)

# Test parameters
window = 15
polyorder = 7
test_orders = [0, 1, 2]

for deriv_order in test_orders
    println("\nDerivative order: $deriv_order")
    println("-" ^ 80)

    # Package version
    try
        result_pkg = savitzky_golay(y_noisy, window, polyorder; deriv=deriv_order, rate=dx)

        # Our version
        our_coeffs = our_sg_coeffs(window, polyorder, deriv_order)
        half_window = window ÷ 2
        result_ours = fill(NaN, n)
        for i = 1:n
            if i <= half_window || i > n - half_window
                continue
            end
            window_data = y_noisy[(i-half_window):(i+half_window)]
            result_ours[i] = dot(our_coeffs, window_data) / (dx^deriv_order)
        end

        # Compare on interior points
        valid_range = (half_window+1):(n-half_window)
        diff_vals = result_pkg.y[valid_range] .- result_ours[valid_range]
        max_diff = maximum(abs.(diff_vals))

        println("  Package result length: $(length(result_pkg.y))")
        println("  Our result length: $(length(result_ours))")
        println("  Max difference on interior: $(round(max_diff, sigdigits=4))")

        if max_diff < 1e-10
            println("  ✓ Results match!")
        else
            println("  ✗ Results differ significantly")
            println("  Package sample: $(result_pkg.y[50:52])")
            println("  Our sample: $(result_ours[50:52])")
        end
    catch e
        println("  ERROR: $e")
    end
end

println("\n" * "=" ^ 80)
println("\nPackage features:")
println("  - Has weighting support: Yes (2nd method signature)")
result_check = savitzky_golay(y_noisy, window, polyorder)
println("  - Returns struct: SGolayResults with fields: $(fieldnames(typeof(result_check)))")
println("  - Result vector: .y field")
println("\nOur implementation:")
println("  - Has weighting support: No")
println("  - Handles boundaries: NaN + interpolation")
println("  - Returns: Raw vector")
