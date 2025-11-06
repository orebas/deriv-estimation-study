"""
Test different ways to fit data points with ApproxFun

The issue: When we fit too many coefficients, we overfit noise.
Need to find the right way to fit data with controlled polynomial degree.
"""

using ApproxFun
using Statistics

println("=" ^ 80)
println("ApproxFun Data Fitting Test")
println("=" ^ 80)

# Create test data: sin(2πt) with slight noise
n = 101
t = range(0, 1, length=n)
omega = 2 * π
y_true = sin.(omega .* t)
y_noisy = y_true .+ 1e-3 .* randn(n)

test_point = 0.5
truth_order1 = omega * cos(omega * test_point)

println("\nTest signal: y = sin(2πt)")
println("  N = $n points")
println("  Noise level = 1e-3")
println("  Test point: t = $test_point")
println("  True derivative at t=0.5: $(round(truth_order1, sigdigits=6))")

# Method 1: Fit function directly (WORKS for clean functions)
println("\n" * "-" ^ 80)
println("Method 1: Fit function directly")
println("-" ^ 80)

f_direct = Fun(x -> sin(omega * x), 0..1)
val_0 = f_direct(test_point)
val_1 = f_direct'(test_point)

println("Order 0: $(round(val_0, sigdigits=6)) | Error: $(abs(val_0 - sin(omega * test_point)))")
println("Order 1: $(round(val_1, sigdigits=6)) | Error: $(abs(val_1 - truth_order1))")
println("Coefficients: $(length(coefficients(f_direct)))")

# Method 2: Use points constructor with low degree
println("\n" * "-" ^ 80)
println("Method 2: Fit data with limited degree")
println("-" ^ 80)

# Try different polynomial degrees
for max_degree in [10, 20, 30, 50]
    S = Chebyshev(0..1)

    # Get first max_degree+1 Chebyshev points
    cheb_points = points(S, max_degree + 1)

    # Interpolate our data to these Chebyshev points
    using Interpolations
    itp = linear_interpolation(collect(t), y_noisy)
    y_at_cheb = [itp(clamp(p, 0, 1)) for p in cheb_points]

    # Create Fun from values at Chebyshev points
    f_fit = Fun(S, y_at_cheb)

    val_0 = f_fit(test_point)
    val_1 = f_fit'(test_point)

    error_0 = abs(val_0 - sin(omega * test_point))
    error_1 = abs(val_1 - truth_order1)

    println("  Degree $max_degree: Order 0 error = $(round(error_0, sigdigits=4)) | Order 1 error = $(round(error_1, sigdigits=4)) | Coeffs = $(length(coefficients(f_fit)))")
end

# Method 3: Use least squares fit with specified degree
println("\n" * "-" ^ 80)
println("Method 3: Least squares fit with controlled degree")
println("-" ^ 80)

for max_degree in [10, 20, 30, 50]
    S = Chebyshev(0..1)

    # Create basis functions
    basis = [Fun(S, [zeros(k-1); 1.0]) for k in 1:max_degree+1]

    # Least squares fit
    # Evaluate basis at our data points
    A = zeros(length(t), max_degree + 1)
    for (i, ti) in enumerate(t)
        for (j, φ) in enumerate(basis)
            A[i, j] = φ(ti)
        end
    end

    # Solve least squares
    coeffs = A \ y_noisy

    # Create Fun from coefficients
    f_fit = Fun(S, coeffs)

    val_0 = f_fit(test_point)
    val_1 = f_fit'(test_point)

    error_0 = abs(val_0 - sin(omega * test_point))
    error_1 = abs(val_1 - truth_order1)

    println("  Degree $max_degree: Order 0 error = $(round(error_0, sigdigits=4)) | Order 1 error = $(round(error_1, sigdigits=4)) | Coeffs = $(length(coefficients(f_fit)))")
end

# Method 4: What we're currently doing (BAD - overfits)
println("\n" * "-" ^ 80)
println("Method 4: Current approach (densify to 1000 then transform)")
println("-" ^ 80)

# Densify to 1000 points
using Interpolations
itp = cubic_spline_interpolation(t, y_noisy)
t_dense = range(0, 1, length=1000)
y_dense = [itp(ti) for ti in t_dense]

S = Chebyshev(0..1)
fitted_func = Fun(S, ApproxFun.transform(S, y_dense))

val_0 = fitted_func(test_point)
val_1 = fitted_func'(test_point)

error_0 = abs(val_0 - sin(omega * test_point))
error_1 = abs(val_1 - truth_order1)

println("  1000 points: Order 0 error = $(round(error_0, sigdigits=4)) | Order 1 error = $(round(error_1, sigdigits=4)) | Coeffs = $(length(coefficients(fitted_func)))")

println("\n" * "=" ^ 80)
println("CONCLUSION")
println("=" ^ 80)
println("""
We need to control the polynomial degree!

Best approach: Least squares fit with degree 20-30
- Balances smoothing and accuracy
- Doesn't overfit noise
- Derivatives remain stable

Current approach (1000 coefficients) is catastrophically bad.
""")
