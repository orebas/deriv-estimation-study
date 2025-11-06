"""
Test ApproxFun for derivative computation

ApproxFun is a sophisticated package that automatically selects
Chebyshev basis and handles derivatives elegantly.
"""

using ApproxFun
using TaylorDiff

include("methods/julia/common.jl")

println("=" ^ 80)
println("ApproxFun Derivative Test")
println("=" ^ 80)

# Test on e^x
n = 101
t = range(0, 1, length=n)
y = exp.(t)

test_point = 0.5
truth = exp(test_point)

println("\nTest signal: y = e^x on [0, 1]")
println("Ground truth (all orders): $(round(truth, sigdigits=10))")

# Fit with ApproxFun
println("\n" * "-" ^ 80)
println("ApproxFun Fit")
println("-" ^ 80)

# Create a Fun (ApproxFun's function approximation)
# Option 1: Fit the actual function (best for clean signal)
f_fit = Fun(exp, 0..1)  # Fit e^x on [0,1]

# Option 2: Interpolate data points
# f_fit = Fun(0..1, y, Chebyshev(0..1))  # Fit data to Chebyshev basis

println("ApproxFun function created")
println("  Basis: Chebyshev")
println("  Number of coefficients: $(length(coefficients(f_fit)))")

# Compute derivatives
println("\nOrder | Value      | Error     | Rel Error | Method")
println("-" ^ 70)

for order in 0:7
    # ApproxFun has a derivative operator
    if order == 0
        val_approx = f_fit(test_point)
    else
        # Compute nth derivative
        f_deriv = f_fit
        for i in 1:order
            f_deriv = f_deriv'  # ApproxFun overloads '
        end
        val_approx = f_deriv(test_point)
    end

    error = abs(val_approx - truth)
    rel_error = error / abs(truth)

    println("  $order   | $(round(val_approx, sigdigits=6)) | $(round(error, sigdigits=4)) | $(round(rel_error, sigdigits=4)) | ApproxFun")
end

# Compare with our methods
println("\n" * "=" ^ 80)
println("Comparison: ApproxFun vs Manual Chebyshev vs AAA")
println("=" ^ 80)

# Manual Chebyshev (from previous test)
function fit_chebyshev_simple(x, y, degree)
    x_min, x_max = extrema(x)
    x_scaled = @. 2 * (x - x_min) / (x_max - x_min) - 1
    n = length(x)
    V = zeros(n, degree + 1)
    for i in 1:n
        V[i, 1] = 1.0
        if degree >= 1
            V[i, 2] = x_scaled[i]
        end
        for k in 3:(degree + 1)
            V[i, k] = 2 * x_scaled[i] * V[i, k-1] - V[i, k-2]
        end
    end
    coeffs = V \ y
    return coeffs, x_min, x_max
end

function eval_chebyshev_simple(coeffs, x, x_min, x_max)
    x_scaled = 2 * (x - x_min) / (x_max - x_min) - 1
    degree = length(coeffs) - 1
    if degree == 0
        return coeffs[1]
    end
    T_prev = 1.0
    T_curr = x_scaled
    result = coeffs[1] * T_prev + coeffs[2] * T_curr
    for k in 2:degree
        T_next = 2 * x_scaled * T_curr - T_prev
        result += coeffs[k + 1] * T_next
        T_prev = T_curr
        T_curr = T_next
    end
    return result
end

using ForwardDiff

coeffs_cheby, x_min, x_max = fit_chebyshev_simple(collect(t), y, 30)
cheby_func = x -> eval_chebyshev_simple(coeffs_cheby, x, x_min, x_max)

# AAA
fitted_aaa = fit_aaa(collect(t), y; tol=1e-13, mmax=100)

# Comparison table
println("\nOrder | ApproxFun Err | Manual Cheby Err | AAA Error | Winner")
println("-" ^ 80)

for order in 0:7
    # ApproxFun
    if order == 0
        val_approxfun = f_fit(test_point)
    else
        f_deriv = f_fit
        for i in 1:order
            f_deriv = f_deriv'
        end
        val_approxfun = f_deriv(test_point)
    end
    approxfun_error = abs(val_approxfun - truth)

    # Manual Chebyshev
    if order == 0
        val_cheby = cheby_func(test_point)
    else
        f_cheby = cheby_func
        for i in 1:order
            f_prev = f_cheby
            f_cheby = x -> ForwardDiff.derivative(f_prev, x)
        end
        val_cheby = f_cheby(test_point)
    end
    cheby_error = abs(val_cheby - truth)

    # AAA
    if order == 0
        val_aaa = fitted_aaa(test_point)
    else
        val_aaa = TaylorDiff.derivative(fitted_aaa, test_point, Val(order))
    end
    aaa_error = abs(val_aaa - truth)

    # Determine winner
    errors = [approxfun_error, cheby_error, aaa_error]
    min_error = minimum(errors)

    if min_error == approxfun_error
        winner = "ApproxFun ✓"
    elseif min_error == cheby_error
        winner = "Chebyshev ✓"
    else
        winner = "AAA"
    end

    println("  $order   | $(round(approxfun_error, sigdigits=4)) | $(round(cheby_error, sigdigits=4)) | $(round(aaa_error, sigdigits=4)) | $winner")
end

println("\n" * "=" ^ 80)
println("CONCLUSION")
println("=" ^ 80)
println("""
ApproxFun advantages:
✓ Automatic adaptive basis selection
✓ Built-in derivative operator (f')
✓ Very clean API
✓ Optimized Chebyshev transforms (FFT)
✓ Handles domains other than [-1,1]

If ApproxFun performs well, we should use it for the ROM!
""")
