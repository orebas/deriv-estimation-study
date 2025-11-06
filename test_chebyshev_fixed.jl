"""
Fixed Chebyshev derivative computation

The issue: coordinate scaling in derivative needs to be handled correctly.
"""

using LinearAlgebra
using FFTW

println("=" ^ 80)
println("Fixed Chebyshev Derivatives")
println("=" ^ 80)

# Simple approach: use ForwardDiff on the Chebyshev evaluator
using ForwardDiff

"""Fit Chebyshev series (least squares)"""
function fit_chebyshev_simple(x, y, degree)
    # Map to [-1, 1]
    x_min, x_max = extrema(x)
    x_scaled = @. 2 * (x - x_min) / (x_max - x_min) - 1

    # Vandermonde matrix
    n = length(x)
    V = zeros(n, degree + 1)

    for i in 1:n
        V[i, 1] = 1.0  # T₀
        if degree >= 1
            V[i, 2] = x_scaled[i]  # T₁
        end
        for k in 3:(degree + 1)
            # Recurrence: Tₙ₊₁(x) = 2xTₙ(x) - Tₙ₋₁(x)
            V[i, k] = 2 * x_scaled[i] * V[i, k-1] - V[i, k-2]
        end
    end

    coeffs = V \ y
    return coeffs, x_min, x_max
end

"""Evaluate Chebyshev series at x"""
function eval_chebyshev_simple(coeffs, x, x_min, x_max)
    # Map to [-1, 1]
    x_scaled = 2 * (x - x_min) / (x_max - x_min) - 1

    # Evaluate T₀, T₁, T₂, ... at x_scaled
    degree = length(coeffs) - 1

    if degree == 0
        return coeffs[1]
    end

    # Use recurrence to evaluate
    T_prev = 1.0  # T₀
    T_curr = x_scaled  # T₁

    result = coeffs[1] * T_prev + coeffs[2] * T_curr

    for k in 2:degree
        T_next = 2 * x_scaled * T_curr - T_prev
        result += coeffs[k + 1] * T_next
        T_prev = T_curr
        T_curr = T_next
    end

    return result
end

"""
Create a closure for Chebyshev evaluation that ForwardDiff can differentiate
"""
function make_chebyshev_function(coeffs, x_min, x_max)
    return x -> eval_chebyshev_simple(coeffs, x, x_min, x_max)
end

# Test on e^x
println("\nTest: e^x on [0, 1]")

n = 101
t = range(0, 1, length=n)
y = exp.(t)

test_point = 0.5
truth = exp(test_point)

println("Test point: t = $test_point")
println("Ground truth (all orders): $(round(truth, sigdigits=10))")

degrees = [10, 20, 30]

for degree in degrees
    println("\n" * "-" ^ 80)
    println("Chebyshev degree: $degree")
    println("-" ^ 80)

    # Fit
    coeffs, x_min, x_max = fit_chebyshev_simple(collect(t), y, degree)

    # Create evaluator
    cheby_func = make_chebyshev_function(coeffs, x_min, x_max)

    # Test derivatives using ForwardDiff
    println("Order | Value      | Error     | Rel Error")
    println("-" ^ 60)

    for order in 0:7
        if order == 0
            val = cheby_func(test_point)
        elseif order == 1
            val = ForwardDiff.derivative(cheby_func, test_point)
        elseif order == 2
            val = ForwardDiff.derivative(x -> ForwardDiff.derivative(cheby_func, x), test_point)
        elseif order == 3
            val = ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> ForwardDiff.derivative(cheby_func, y), x), test_point)
        else
            # Recursive for higher orders
            f = cheby_func
            for i in 1:order
                f_prev = f
                f = x -> ForwardDiff.derivative(f_prev, x)
            end
            val = f(test_point)
        end

        error = abs(val - truth)
        rel_error = error / abs(truth)

        println("  $order   | $(round(val, sigdigits=6)) | $(round(error, sigdigits=4)) | $(round(rel_error, sigdigits=4))")
    end
end

# Compare with AAA
println("\n" * "=" ^ 80)
println("Comparison: Chebyshev (degree 30) vs AAA")
println("=" ^ 80)

include("methods/julia/common.jl")
using TaylorDiff

coeffs_cheby, x_min, x_max = fit_chebyshev_simple(collect(t), y, 30)
cheby_func = make_chebyshev_function(coeffs_cheby, x_min, x_max)

fitted_aaa = fit_aaa(collect(t), y; tol=1e-13, mmax=100)

println("\nOrder | Chebyshev Error | AAA Error     | Winner")
println("-" ^ 70)

for order in 0:7
    # Chebyshev
    if order == 0
        cheby_val = cheby_func(test_point)
    else
        # Use nested ForwardDiff
        f = cheby_func
        for i in 1:order
            f_prev = f
            f = x -> ForwardDiff.derivative(f_prev, x)
        end
        cheby_val = f(test_point)
    end
    cheby_error = abs(cheby_val - truth)

    # AAA
    if order == 0
        aaa_val = fitted_aaa(test_point)
    else
        aaa_val = TaylorDiff.derivative(fitted_aaa, test_point, Val(order))
    end
    aaa_error = abs(aaa_val - truth)

    winner = cheby_error < aaa_error ? "Chebyshev ✓" : "AAA"

    println("  $order   | $(round(cheby_error, sigdigits=4)) | $(round(aaa_error, sigdigits=4)) | $winner")
end

println("\n" * "=" ^ 80)
println("CONCLUSION")
println("=" ^ 80)
