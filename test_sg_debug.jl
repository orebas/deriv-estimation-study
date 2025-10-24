#!/usr/bin/env julia
"""
Minimal test to debug Savitzky-Golay derivative computation
"""

using ForwardDiff
using TaylorDiff

# Include the extracted modules
include("methods/julia/common.jl")
include("methods/julia/filtering/filters.jl")

# Simple test function: f(x) = x^2
x = range(0, 1, length=51) |> collect
y = x .^ 2

println("="^80)
println("SAVITZKY-GOLAY DEBUG TEST")
println("="^80)
println()

# Test what fit_savitzky_golay returns
fitted = fit_savitzky_golay(x, y; window=11, polyorder=5)

println("Testing fitted function at x=0.5:")
println("  f(0.5) = ", fitted(0.5))
println("  Expected f(0.5) = 0.25")
println()

# Test nth_deriv_taylor directly
println("Testing nth_deriv_taylor on the fitted function:")
for order in 0:3
    try
        val = nth_deriv_taylor(fitted, order, 0.5)
        println("  Order $order at x=0.5: ", val)
    catch e
        println("  Order $order at x=0.5: ERROR - ", e)
    end
end
println()

# Expected derivatives of f(x) = x^2:
# f(x) = x^2          → f(0.5) = 0.25
# f'(x) = 2x          → f'(0.5) = 1.0
# f''(x) = 2          → f''(0.5) = 2.0
# f'''(x) = 0         → f'''(0.5) = 0.0

println("Expected values for f(x) = x^2:")
println("  Order 0 at x=0.5: 0.25")
println("  Order 1 at x=0.5: 1.0")
println("  Order 2 at x=0.5: 2.0")
println("  Order 3 at x=0.5: 0.0")
println()

# Now test the full evaluate_savitzky_golay function
println("Testing evaluate_savitzky_golay:")
x_eval = [0.5]
orders = [0, 1, 2, 3]
result = evaluate_savitzky_golay(x, y, x_eval, orders)

println("Result success: ", result.success)
println("Predictions:")
for order in orders
    if haskey(result.predictions, order)
        println("  Order $order: ", result.predictions[order])
    else
        println("  Order $order: MISSING")
    end
end
println()
println("Failures:")
for (order, msg) in result.failures
    println("  Order $order: ", msg)
end
println()
println("="^80)
