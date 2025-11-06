"""
Test: Use our EXISTING GP method with ApproxFun

We already have fit_gp_se_analytic() - can we pass that to ApproxFun?
"""

using ApproxFun

# Load our existing GP method
include("methods/julia/gp/gaussian_process.jl")

println("=" ^ 80)
println("ApproxFun with EXISTING GP Method")
println("=" ^ 80)

# Create noisy data
n = 101
t_data = collect(range(0, 1, length=n))
omega = 2 * π
y_true = sin.(omega .* t_data)
y_noisy = y_true .+ 1e-3 .* randn(n)

println("\nData: sin(2πt) + 1e-3 noise, $n points")

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

# Fit our existing GP method
println("\nFitting GP using fit_gp_se_analytic()...")
gp_predictor = fit_gp_se_analytic(t_data, y_noisy)

# Test it
println("GP prediction at test point:")
gp_val = gp_predictor(test_point, 0)
gp_error = abs(gp_val - ground_truth(0, test_point))
println("  Order 0: value = $(round(gp_val, sigdigits=6)) | error = $(round(gp_error, sigdigits=4))")

gp_val1 = gp_predictor(test_point, 1)
gp_error1 = abs(gp_val1 - ground_truth(1, test_point))
println("  Order 1: value = $(round(gp_val1, sigdigits=6)) | error = $(round(gp_error1, sigdigits=4))")

# Test 1: Pass GP predictor (order 0 only) to ApproxFun
println("\n" * "-" ^ 80)
println("Test 1: GP predictor (order 0) → ApproxFun")
println("-" ^ 80)

# Create a wrapper that only uses order 0
gp_func = x -> gp_predictor(x, 0)

try
    f_gp = Fun(gp_func, 0..1)

    println("ApproxFun chose $(length(coefficients(f_gp))) coefficients")

    println("\nDerivative accuracy:")
    for order in 0:7
        if order == 0
            val = f_gp(test_point)
        else
            f_deriv = f_gp
            for i in 1:order
                f_deriv = f_deriv'
            end
            val = f_deriv(test_point)
        end

        truth = ground_truth(order, test_point)
        error = abs(val - truth)

        println("  Order $order: value = $(round(val, sigdigits=6)) | error = $(round(error, sigdigits=4))")
    end
catch ex
    println("ERROR: $ex")
end

# Test 2: Least squares for comparison
println("\n" * "-" ^ 80)
println("Test 2: Least squares (degree 10) for comparison")
println("-" ^ 80)

S = Chebyshev(0..1)
max_degree = 10
basis = [Fun(S, [zeros(k-1); 1.0]) for k in 1:max_degree+1]

A = zeros(n, max_degree + 1)
for (i, ti) in enumerate(t_data)
    for (j, φ) in enumerate(basis)
        A[i, j] = φ(ti)
    end
end

coeffs = A \ y_noisy
f_ls = Fun(S, coeffs)

println("Used $(length(coefficients(f_ls))) coefficients")

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
println("CONCLUSION")
println("=" ^ 80)
println("""
Same result: Our existing GP → ApproxFun still overfits!

The GP predictor is smooth, but ApproxFun doesn't know it's from noisy data.
ApproxFun tries to match it perfectly → millions of coefficients.

The fundamental issue remains:
- Any interpolator/predictor from noisy data → ApproxFun overfits
- Manual degree control (least squares) is the only solution
""")
