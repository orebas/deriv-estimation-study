"""
Test AAA differentiation on CLEAN signal (no noise)

Test: y = e^x on [0, 1]
All derivatives of e^x equal e^x, so this is a perfect test case.
"""

using BaryRational
using TaylorDiff
using LinearAlgebra

# Include common utilities
include("methods/julia/common.jl")

println("=" ^ 80)
println("AAA Differentiation Test on Clean Signal")
println("=" ^ 80)

# Test signal: e^x on [0, 1]
n_points = 101
t = range(0, 1, length=n_points)
y_clean = exp.(t)

println("\nTest signal: y = e^x")
println("  Domain: [0, 1]")
println("  Points: $n_points")
println("  All derivatives should equal e^x")

# Ground truth derivatives (all equal e^x for exponential)
ground_truth = Dict(order => exp.(t) for order in 0:7)

# Test different AAA tolerances
tolerances = [1e-13, 1e-10, 1e-8, 1e-6]

for tol in tolerances
    println("\n" * "=" ^ 80)
    println("AAA Tolerance: $tol")
    println("=" ^ 80)

    # Fit AAA
    fitted = fit_aaa(collect(t), y_clean; tol=tol, mmax=100)

    # Compute derivatives
    for order in 0:7
        try
            # Compute derivatives at all points
            deriv = [nth_deriv_taylor(fitted, order, ti) for ti in t]

            # Compute error vs ground truth
            errors = abs.(deriv .- ground_truth[order])
            rmse = sqrt(mean(errors.^2))
            max_error = maximum(errors)
            mean_error = mean(errors)

            # Check if all finite
            n_finite = sum(isfinite.(deriv))

            if n_finite == length(deriv)
                println("  Order $order: RMSE = $(round(rmse, sigdigits=4)), Max = $(round(max_error, sigdigits=4)), Mean = $(round(mean_error, sigdigits=4))")
            else
                println("  Order $order: $n_finite/$(length(deriv)) finite (some NaN/Inf)")
            end

        catch e
            println("  Order $order: FAILED - $e")
        end
    end
end

# Test with different grid sizes
println("\n" * "=" ^ 80)
println("Grid Size Test (tol=1e-13)")
println("=" ^ 80)

grid_sizes = [51, 101, 201, 501, 1001]

for n in grid_sizes
    println("\n$n points:")

    t_grid = range(0, 1, length=n)
    y_grid = exp.(t_grid)

    # Fit AAA
    fitted = fit_aaa(collect(t_grid), y_grid; tol=1e-13, mmax=100)

    # Test derivatives at evaluation points (use original 101 points)
    t_eval = range(0, 1, length=101)

    for order in [0, 1, 2, 4, 7]
        try
            deriv = [nth_deriv_taylor(fitted, order, ti) for ti in t_eval]
            truth = exp.(t_eval)

            errors = abs.(deriv .- truth)
            rmse = sqrt(mean(errors.^2))

            println("  Order $order: RMSE = $(round(rmse, sigdigits=4))")

        catch e
            println("  Order $order: FAILED")
        end
    end
end

println("\n" * "=" ^ 80)
println("CONCLUSION")
println("=" ^ 80)
println("""
If AAA works well on e^x (clean signal), then the problem with PyNumDiff
is noise amplification, not AAA differentiation itself.

If AAA fails even on e^x, then we have a fundamental issue with how
we're differentiating the rational approximation.
""")
