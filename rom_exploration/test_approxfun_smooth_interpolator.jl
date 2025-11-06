"""
Test: Pass SMOOTHED data interpolator to ApproxFun

Key insight: If we interpolate ALREADY-SMOOTHED data (from PyNumDiff, etc.),
ApproxFun might pick a reasonable degree automatically!
"""

using ApproxFun
using Interpolations
using Statistics

println("=" ^ 80)
println("ApproxFun with SMOOTHED Data Interpolator")
println("=" ^ 80)

# Create noisy data
n = 101
t = range(0, 1, length=n)
omega = 2 * π
y_true = sin.(omega .* t)
y_noisy = y_true .+ 1e-3 .* randn(n)

# Simulate what PyNumDiff does: SMOOTH the signal
# Use a simple moving average for this test
function smooth_signal(y, window=11)
    n = length(y)
    y_smooth = copy(y)
    half_win = window ÷ 2

    for i in 1:n
        i_start = max(1, i - half_win)
        i_end = min(n, i + half_win)
        y_smooth[i] = mean(y[i_start:i_end])
    end

    return y_smooth
end

y_smooth = smooth_signal(y_noisy, 11)

println("\nOriginal data: sin(2πt) + 1e-3 noise")
println("After smoothing: moving average (window=11)")
println("Points: $n")

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

# Test 1: Cubic spline of NOISY data → ApproxFun
println("\n" * "-" ^ 80)
println("Test 1: Cubic spline of NOISY data → ApproxFun")
println("-" ^ 80)

itp_noisy = cubic_spline_interpolation(t, y_noisy)
f_noisy = Fun(itp_noisy, 0..1)

println("ApproxFun chose $(length(coefficients(f_noisy))) coefficients")

errors_noisy = []
for order in 0:7
    if order == 0
        val = f_noisy(test_point)
    else
        f_deriv = f_noisy
        for i in 1:order
            f_deriv = f_deriv'
        end
        val = f_deriv(test_point)
    end

    truth = ground_truth(order, test_point)
    error = abs(val - truth)
    push!(errors_noisy, error)

    println("  Order $order: error = $(round(error, sigdigits=4))")
end

# Test 2: Cubic spline of SMOOTHED data → ApproxFun
println("\n" * "-" ^ 80)
println("Test 2: Cubic spline of SMOOTHED data → ApproxFun")
println("-" ^ 80)

itp_smooth = cubic_spline_interpolation(t, y_smooth)
f_smooth = Fun(itp_smooth, 0..1)

println("ApproxFun chose $(length(coefficients(f_smooth))) coefficients")

errors_smooth = []
for order in 0:7
    if order == 0
        val = f_smooth(test_point)
    else
        f_deriv = f_smooth
        for i in 1:order
            f_deriv = f_deriv'
        end
        val = f_deriv(test_point)
    end

    truth = ground_truth(order, test_point)
    error = abs(val - truth)
    push!(errors_smooth, error)

    println("  Order $order: error = $(round(error, sigdigits=4))")
end

# Test 3: Linear interpolation of SMOOTHED data → ApproxFun
println("\n" * "-" ^ 80)
println("Test 3: Linear interpolation of SMOOTHED data → ApproxFun")
println("-" ^ 80)

itp_linear_smooth = linear_interpolation(t, y_smooth)
f_linear_smooth = Fun(itp_linear_smooth, 0..1)

println("ApproxFun chose $(length(coefficients(f_linear_smooth))) coefficients")

errors_linear_smooth = []
for order in 0:7
    if order == 0
        val = f_linear_smooth(test_point)
    else
        f_deriv = f_linear_smooth
        for i in 1:order
            f_deriv = f_deriv'
        end
        val = f_deriv(test_point)
    end

    truth = ground_truth(order, test_point)
    error = abs(val - truth)
    push!(errors_linear_smooth, error)

    println("  Order $order: error = $(round(error, sigdigits=4))")
end

# Test 4: Our least squares (degree 10) for comparison
println("\n" * "-" ^ 80)
println("Test 4: Least squares (degree 10) on SMOOTHED data")
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

coeffs = A \ y_smooth
f_ls = Fun(S, coeffs)

println("Used $(length(coefficients(f_ls))) coefficients")

errors_ls = []
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
    push!(errors_ls, error)

    println("  Order $order: error = $(round(error, sigdigits=4))")
end

# Comparison table
println("\n" * "=" ^ 80)
println("COMPARISON")
println("=" ^ 80)

println("\nMethod                              | Coeffs | Order 1 | Order 3 | Order 7")
println("-" ^ 85)
println("Cubic spline (noisy) → ApproxFun    | $(lpad(length(coefficients(f_noisy)), 6)) | $(lpad(round(errors_noisy[2], sigdigits=4), 7)) | $(lpad(round(errors_noisy[4], sigdigits=4), 7)) | $(lpad(round(errors_noisy[8], sigdigits=4), 7))")
println("Cubic spline (smooth) → ApproxFun   | $(lpad(length(coefficients(f_smooth)), 6)) | $(lpad(round(errors_smooth[2], sigdigits=4), 7)) | $(lpad(round(errors_smooth[4], sigdigits=4), 7)) | $(lpad(round(errors_smooth[8], sigdigits=4), 7))")
println("Linear interp (smooth) → ApproxFun  | $(lpad(length(coefficients(f_linear_smooth)), 6)) | $(lpad(round(errors_linear_smooth[2], sigdigits=4), 7)) | $(lpad(round(errors_linear_smooth[4], sigdigits=4), 7)) | $(lpad(round(errors_linear_smooth[8], sigdigits=4), 7))")
println("Least squares deg 10 (smooth)       | $(lpad(length(coefficients(f_ls)), 6)) | $(lpad(round(errors_ls[2], sigdigits=4), 7)) | $(lpad(round(errors_ls[4], sigdigits=4), 7)) | $(lpad(round(errors_ls[8], sigdigits=4), 7))")

println("\n" * "=" ^ 80)
println("QUESTION")
println("=" ^ 80)
println("""
Does pre-smoothing the data help ApproxFun pick a better degree?

If cubic spline of SMOOTHED data gives reasonable coefficients,
we could use this workflow:
  1. Python: Run method, get smoothed signal
  2. Python: Save smoothed signal (NO densification)
  3. Julia: Build interpolator from smoothed signal
  4. Julia: Pass interpolator to ApproxFun → automatic degree selection!
  5. Julia: Compute derivatives

Let's see if it works!
""")
