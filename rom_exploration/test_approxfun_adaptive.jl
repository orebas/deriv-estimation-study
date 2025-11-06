"""
Test ApproxFun's automatic degree selection for DATA FITTING

The question: Can we let ApproxFun pick the degree automatically when fitting data points?
"""

using ApproxFun

println("=" ^ 80)
println("ApproxFun Automatic Degree Selection for Data")
println("=" ^ 80)

# Create test data: sin(2πt) with noise
n = 101
t = range(0, 1, length=n)
omega = 2 * π
y_true = sin.(omega .* t)
y_noisy = y_true .+ 1e-3 .* randn(n)

test_point = 0.5
truth_order1 = omega * cos(omega * test_point)

println("\nTest: sin(2πt) with 1e-3 noise, $n points")

# Method 1: What ApproxFun does for functions
println("\n" * "-" ^ 80)
println("Method 1: Fit function directly (ApproxFun automatic)")
println("-" ^ 80)

f_auto = Fun(x -> sin(omega * x), 0..1)
println("Coefficients chosen: $(length(coefficients(f_auto)))")
println("Order 1 error: $(abs(f_auto'(test_point) - truth_order1))")

# Method 2: Try to fit data using Fun with space
println("\n" * "-" ^ 80)
println("Method 2: Fit data to Chebyshev space (what does ApproxFun do?)")
println("-" ^ 80)

S = Chebyshev(0..1)

# Try different approaches from ApproxFun
# Approach A: Use points() to get Chebyshev points
println("\nApproach A: Interpolate at Chebyshev points")
for n_cheb in [10, 20, 30, 50, 101]
    cheb_pts = points(S, n_cheb)

    # Interpolate our noisy data to these points
    using Interpolations
    itp = linear_interpolation(collect(t), y_noisy)
    y_at_cheb = [itp(clamp(p, 0, 1)) for p in cheb_pts]

    # Create Fun - this should use these values
    f_fit = Fun(S, y_at_cheb)

    error_0 = abs(f_fit(test_point) - sin(omega * test_point))
    error_1 = abs(f_fit'(test_point) - truth_order1)
    n_coeffs = length(coefficients(f_fit))

    println("  $n_cheb Cheb points → $n_coeffs coeffs | Order 1 error: $(round(error_1, sigdigits=4))")
end

# Approach B: Use transform but with fewer points
println("\nApproach B: transform() with downsampled data")
for downsample in [2, 5, 10, 20, 50]
    indices = 1:downsample:n
    t_down = t[indices]
    y_down = y_noisy[indices]

    n_down = length(t_down)

    if n_down < 3
        continue
    end

    S_down = Chebyshev(0..1)

    # Use transform - but this expects data at Chebyshev points
    # This will probably fail or give wrong results
    try
        f_fit = Fun(S_down, ApproxFun.transform(S_down, y_down))

        error_1 = abs(f_fit'(test_point) - truth_order1)
        n_coeffs = length(coefficients(f_fit))

        println("  Downsample $downsample → $n_down points → $n_coeffs coeffs | Order 1 error: $(round(error_1, sigdigits=4))")
    catch e
        println("  Downsample $downsample → FAILED: $e")
    end
end

# Approach C: Least squares but let ApproxFun determine truncation
println("\nApproach C: Least squares with coefficient truncation")

using Statistics

for max_degree in [10, 20, 30, 50]
    # Do least squares fit
    basis = [Fun(S, [zeros(k-1); 1.0]) for k in 1:max_degree+1]
    A = zeros(n, max_degree + 1)
    for (i, ti) in enumerate(t)
        for (j, φ) in enumerate(basis)
            A[i, j] = φ(ti)
        end
    end

    coeffs = A \ y_noisy

    # Find truncation point where coefficients become negligible
    # ApproxFun typically uses relative tolerance ~1e-14 or adaptive
    abs_coeffs = abs.(coeffs)
    max_coeff = maximum(abs_coeffs)

    # Try different tolerance levels
    for tol in [1e-6, 1e-8, 1e-10]
        # Find where to truncate
        truncate_at = findlast(c -> c > tol * max_coeff, abs_coeffs)

        if truncate_at === nothing || truncate_at < 3
            truncate_at = 3
        end

        coeffs_truncated = coeffs[1:truncate_at]

        f_fit = Fun(S, coeffs_truncated)

        error_1 = abs(f_fit'(test_point) - truth_order1)

        println("  Deg $max_degree, tol $tol → $(length(coeffs_truncated)) coeffs | Order 1 error: $(round(error_1, sigdigits=4))")
    end
end

println("\n" * "=" ^ 80)
println("QUESTION")
println("=" ^ 80)
println("""
What's the RIGHT way to fit data points with ApproxFun?

Option 1: Manually control degree (what we're doing)
  - Requires tuning max_degree parameter
  - No automatic adaptation

Option 2: Use Chebyshev points interpolation
  - Requires choosing number of points
  - Still manual tuning

Option 3: Least squares + automatic truncation
  - Fit high degree, truncate small coefficients
  - More like what ApproxFun does for functions

Option 4: Check ApproxFun docs for a built-in data fitting function?
""")
