"""
Compare different methods for computing derivatives of AAA approximants

Methods:
1. TaylorDiff (current approach)
2. ForwardDiff
3. Finite differences on AAA
4. Analytic formula for barycentric derivative
"""

using BaryRational
using TaylorDiff
using ForwardDiff
using LinearAlgebra

include("methods/julia/common.jl")

println("=" ^ 80)
println("Comparing AAA Derivative Methods")
println("=" ^ 80)

# Test on e^x
n = 101
t = range(0, 1, length=n)
y = exp.(t)

fitted = fit_aaa(collect(t), y; tol=1e-13, mmax=100)

println("\nTest signal: y = e^x on [0, 1]")
println("Ground truth: all derivatives equal e^x")

# Test point
test_point = 0.5
truth = exp(test_point)

println("\nTest point: t = $test_point")
println("Ground truth: e^0.5 = $(round(truth, sigdigits=10))")

# Method 1: TaylorDiff (current)
println("\n" * "=" ^ 80)
println("Method 1: TaylorDiff (current approach)")
println("=" ^ 80)

for order in 0:7
    try
        if order == 0
            deriv = fitted(test_point)
        else
            deriv = TaylorDiff.derivative(fitted, test_point, Val(order))
        end
        error = abs(deriv - truth)
        rel_error = error / abs(truth)
        println("  Order $order: value = $(round(deriv, sigdigits=6)), error = $(round(error, sigdigits=4)), rel = $(round(rel_error, sigdigits=4))")
    catch e
        println("  Order $order: FAILED - $e")
    end
end

# Method 2: ForwardDiff (recursive)
println("\n" * "=" ^ 80)
println("Method 2: ForwardDiff (recursive)")
println("=" ^ 80)

function nth_deriv_forward(f, n::Int, t)
    n == 0 && return f(t)
    n == 1 && return ForwardDiff.derivative(f, t)
    g(x) = nth_deriv_forward(f, n - 1, x)
    return ForwardDiff.derivative(g, t)
end

for order in 0:7
    try
        deriv = nth_deriv_forward(fitted, order, test_point)
        error = abs(deriv - truth)
        rel_error = error / abs(truth)
        println("  Order $order: value = $(round(deriv, sigdigits=6)), error = $(round(error, sigdigits=4)), rel = $(round(rel_error, sigdigits=4))")
    catch e
        println("  Order $order: FAILED - $e")
    end
end

# Method 3: Finite differences on AAA
println("\n" * "=" ^ 80)
println("Method 3: Finite Differences on AAA")
println("=" ^ 80)

function finite_diff_deriv(f, n::Int, t, h=1e-5)
    if n == 0
        return f(t)
    elseif n == 1
        # Central difference: f'(t) ≈ [f(t+h) - f(t-h)] / (2h)
        return (f(t + h) - f(t - h)) / (2h)
    else
        # Recursive: differentiate the (n-1)th derivative
        g(x) = finite_diff_deriv(f, n - 1, x, h)
        return (g(t + h) - g(t - h)) / (2h)
    end
end

for h in [1e-3, 1e-5, 1e-7]
    println("\n  Step size h = $h:")
    for order in [0, 1, 2, 4, 7]
        try
            deriv = finite_diff_deriv(fitted, order, test_point, h)
            error = abs(deriv - truth)
            rel_error = error / abs(truth)
            println("    Order $order: value = $(round(deriv, sigdigits=6)), error = $(round(error, sigdigits=4)), rel = $(round(rel_error, sigdigits=4))")
        catch e
            println("    Order $order: FAILED")
        end
    end
end

# Method 4: Analytic first derivative of barycentric form
println("\n" * "=" ^ 80)
println("Method 4: Analytic Barycentric Derivative (1st order only)")
println("=" ^ 80)

function bary_first_derivative(f_vals::Vector{T}, x_pts::Vector{T}, w::Vector{T}, z) where {T}
    # Compute r(z) = num(z) / den(z)
    # r'(z) = [num'(z)*den(z) - num(z)*den'(z)] / den(z)²

    num = zero(T)
    den = zero(T)
    num_deriv = zero(T)
    den_deriv = zero(T)

    for j in eachindex(f_vals)
        term = w[j] / (z - x_pts[j])
        num += term * f_vals[j]
        den += term

        # Derivative: d/dz [w/(z-x)] = -w/(z-x)²
        deriv_term = -w[j] / (z - x_pts[j])^2
        num_deriv += deriv_term * f_vals[j]
        den_deriv += deriv_term
    end

    # Quotient rule: (u/v)' = (u'v - uv')/v²
    return (num_deriv * den - num * den_deriv) / den^2
end

try
    # Order 0: function value
    val = fitted(test_point)
    error = abs(val - truth)
    println("  Order 0: value = $(round(val, sigdigits=6)), error = $(round(error, sigdigits=4))")

    # Order 1: analytic derivative
    deriv1 = bary_first_derivative(fitted.f, fitted.x, fitted.w, test_point)
    error = abs(deriv1 - truth)
    rel_error = error / abs(truth)
    println("  Order 1: value = $(round(deriv1, sigdigits=6)), error = $(round(error, sigdigits=4)), rel = $(round(rel_error, sigdigits=4))")

    # Compare with TaylorDiff
    deriv1_taylor = TaylorDiff.derivative(fitted, test_point, Val(1))
    println("  Order 1 (TaylorDiff): value = $(round(deriv1_taylor, sigdigits=6))")
    println("  Difference: $(abs(deriv1 - deriv1_taylor))")

catch e
    println("  FAILED - $e")
end

println("\n" * "=" ^ 80)
println("CONCLUSION")
println("=" ^ 80)
println("""
If finite differences work better than AD, it suggests the issue is with
how AD differentiates the rational function (quotient rule amplification).

If analytic formula for 1st derivative works better, we could potentially
derive stable formulas for higher orders.

If all methods fail similarly, the issue is fundamental to the rational
approximation itself.
""")
