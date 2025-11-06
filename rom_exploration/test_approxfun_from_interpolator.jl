"""
Test: Pass an interpolator function to ApproxFun

Can we let ApproxFun pick the degree automatically by giving it
an interpolating function built from our data points?
"""

using ApproxFun
using Interpolations
using Statistics

println("=" ^ 80)
println("ApproxFun with Interpolator Function")
println("=" ^ 80)

# Test data: sin(2πt) with noise
n = 101
t = range(0, 1, length=n)
omega = 2 * π
y_true = sin.(omega .* t)
y_noisy = y_true .+ 1e-3 .* randn(n)

test_point = 0.5

function ground_truth(order, x)
    if order == 0
        return sin(omega * x)
    elseif order == 1
        return omega * cos(omega * x)
    elseif order == 2
        return -(omega^2) * sin(omega * x)
    elseif order == 3
        return -(omega^3) * cos(omega * x)
    elseif order == 4
        return omega^4 * sin(omega * x)
    elseif order == 5
        return omega^5 * cos(omega * x)
    elseif order == 6
        return -(omega^6) * sin(omega * x)
    elseif order == 7
        return -(omega^7) * cos(omega * x)
    end
end

println("\nTest: sin(2πt) with 1e-3 noise, $n points")
println("Test point: t = $test_point")

# Method 1: Linear interpolation → ApproxFun
println("\n" * "-" ^ 80)
println("Method 1: Linear interpolation → Fun(interpolator, domain)")
println("-" ^ 80)

itp_linear = linear_interpolation(t, y_noisy)
f_linear = Fun(itp_linear, 0..1)

println("ApproxFun chose $(length(coefficients(f_linear))) coefficients")

println("\nDerivative accuracy:")
for order in 0:7
    if order == 0
        val = f_linear(test_point)
    else
        f_deriv = f_linear
        for i in 1:order
            f_deriv = f_deriv'
        end
        val = f_deriv(test_point)
    end

    truth = ground_truth(order, test_point)
    error = abs(val - truth)

    println("  Order $order: value = $(round(val, sigdigits=6)) | error = $(round(error, sigdigits=4))")
end

# Method 2: Cubic spline interpolation → ApproxFun
println("\n" * "-" ^ 80)
println("Method 2: Cubic spline interpolation → Fun(interpolator, domain)")
println("-" ^ 80)

itp_cubic = cubic_spline_interpolation(t, y_noisy)
f_cubic = Fun(itp_cubic, 0..1)

println("ApproxFun chose $(length(coefficients(f_cubic))) coefficients")

println("\nDerivative accuracy:")
for order in 0:7
    if order == 0
        val = f_cubic(test_point)
    else
        f_deriv = f_cubic
        for i in 1:order
            f_deriv = f_deriv'
        end
        val = f_deriv(test_point)
    end

    truth = ground_truth(order, test_point)
    error = abs(val - truth)

    println("  Order $order: value = $(round(val, sigdigits=6)) | error = $(round(error, sigdigits=4))")
end

# Method 3: Skip BSpline (namespace conflict)

# Compare with our manual least squares approach
println("\n" * "-" ^ 80)
println("Method 4: Our least squares (degree 10) for comparison")
println("-" ^ 80)

S = Chebyshev(0..1)
max_degree = 10
basis = [Fun(S, [zeros(k-1); 1.0]) for k in 1:max_degree+1]

A = zeros(n, max_degree + 1)
for (i, ti) in enumerate(t)
    for (j, φ) in enumerate(basis)
        A[i, j] = φ(ti)
    end
end

coeffs = A \ y_noisy
f_ls = Fun(S, coeffs)

println("Used $max_degree degree ($(length(coefficients(f_ls))) coefficients)")

println("\nDerivative accuracy:")
for order in 0:7
    if order == 0
        val = f_ls(test_point)
    else
        f_deriv = f_ls
        for i in 1:order
            f_deriv = f_deriv'
        end
        val = f_deriv(test_point)
    end

    truth = ground_truth(order, test_point)
    error = abs(val - truth)

    println("  Order $order: value = $(round(val, sigdigits=6)) | error = $(round(error, sigdigits=4))")
end

println("\n" * "=" ^ 80)
println("COMPARISON TABLE")
println("=" ^ 80)

println("\nMethod                     | Coeffs  | Order 1 Err | Order 3 Err | Order 7 Err")
println("-" ^ 85)

# We need to recompute since variables are in loop scope
# Linear
itp_linear = linear_interpolation(t, y_noisy)
f_linear = Fun(itp_linear, 0..1)
e1_linear = abs(f_linear'(test_point) - ground_truth(1, test_point))
f3_linear = f_linear
for i in 1:3; global f3_linear = f3_linear'; end
e3_linear = abs(f3_linear(test_point) - ground_truth(3, test_point))
f7_linear = f_linear
for i in 1:7; global f7_linear = f7_linear'; end
e7_linear = abs(f7_linear(test_point) - ground_truth(7, test_point))

println("Linear interp → ApproxFun  | $(length(coefficients(f_linear))) | $(round(e1_linear, sigdigits=4)) | $(round(e3_linear, sigdigits=4)) | $(round(e7_linear, sigdigits=4))")

# Cubic
itp_cubic = cubic_spline_interpolation(t, y_noisy)
f_cubic = Fun(itp_cubic, 0..1)
e1_cubic = abs(f_cubic'(test_point) - ground_truth(1, test_point))
f3_cubic = f_cubic
for i in 1:3; global f3_cubic = f3_cubic'; end
e3_cubic = abs(f3_cubic(test_point) - ground_truth(3, test_point))
f7_cubic = f_cubic
for i in 1:7; global f7_cubic = f7_cubic'; end
e7_cubic = abs(f7_cubic(test_point) - ground_truth(7, test_point))

println("Cubic spline → ApproxFun   | $(length(coefficients(f_cubic))) | $(round(e1_cubic, sigdigits=4)) | $(round(e3_cubic, sigdigits=4)) | $(round(e7_cubic, sigdigits=4))")

# Least squares
S = Chebyshev(0..1)
basis = [Fun(S, [zeros(k-1); 1.0]) for k in 1:11]
A = zeros(n, 11)
for (i, ti) in enumerate(t)
    for (j, φ) in enumerate(basis)
        A[i, j] = φ(ti)
    end
end
coeffs = A \ y_noisy
f_ls = Fun(S, coeffs)
e1_ls = abs(f_ls'(test_point) - ground_truth(1, test_point))
f3_ls = f_ls
for i in 1:3; global f3_ls = f3_ls'; end
e3_ls = abs(f3_ls(test_point) - ground_truth(3, test_point))
f7_ls = f_ls
for i in 1:7; global f7_ls = f7_ls'; end
e7_ls = abs(f7_ls(test_point) - ground_truth(7, test_point))

println("Least squares (deg 10)     | $(length(coefficients(f_ls))) | $(round(e1_ls, sigdigits=4)) | $(round(e3_ls, sigdigits=4)) | $(round(e7_ls, sigdigits=4))")

println("\n" * "=" ^ 80)
println("CONCLUSION")
println("=" ^ 80)
println("""
❌ Passing interpolator to ApproxFun DOES NOT WORK!

Linear interpolator: 2M+ coefficients, order 7 error = 1e52
Cubic spline: 6414 coefficients, order 7 error = 1e20
Least squares (deg 10): 11 coefficients, order 7 error = 6e5 ✓

Why?
- ApproxFun tries to fit the interpolator perfectly
- Interpolator has kinks/artifacts from noisy data
- ApproxFun uses millions of coefficients to match those artifacts
- Derivatives explode catastrophically

Our manual least squares with controlled degree is the RIGHT approach!
The degree parameter IS the smoothing parameter we need for noisy data.
""")
