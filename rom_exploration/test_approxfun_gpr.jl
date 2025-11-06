"""
Test: Pass GPR (Gaussian Process) interpolator to ApproxFun

GPR produces a SMOOTH function (no piecewise artifacts like splines).
Will ApproxFun pick a reasonable degree for a smooth GPR posterior?
"""

using ApproxFun
using GaussianProcesses
using Statistics

println("=" ^ 80)
println("ApproxFun with GPR Interpolator")
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

# Fit GPR
println("\nFitting GPR with SE kernel...")

# Use Squared Exponential (RBF) kernel - infinitely differentiable!
# Length scale controls smoothness
length_scale = 0.1
signal_variance = 1.0
noise_variance = 1e-6  # Small noise for smoothing

# Create kernel
se_kernel = SE(log(length_scale), log(signal_variance))

# Create mean function (zero mean)
mean_func = MeanZero()

# Fit GP
gp = GP(t_data, y_noisy, mean_func, se_kernel, log(noise_variance))

println("GPR fitted!")
println("  Kernel: SE (length_scale=$length_scale)")
println("  Noise variance: $noise_variance")

# Create a function that evaluates the GPR
function gpr_predictor(x)
    # Predict at single point
    μ, σ² = predict_y(gp, [x])
    return μ[1]
end

# Test the GPR directly
println("\nGPR prediction at test point:")
gpr_val = gpr_predictor(test_point)
gpr_error = abs(gpr_val - ground_truth(0, test_point))
println("  Value: $(round(gpr_val, sigdigits=6))")
println("  Error: $(round(gpr_error, sigdigits=4))")

# Test 1: Pass GPR function to ApproxFun
println("\n" * "-" ^ 80)
println("Test 1: GPR function → ApproxFun (automatic degree)")
println("-" ^ 80)

f_gpr = Fun(gpr_predictor, 0..1)

println("ApproxFun chose $(length(coefficients(f_gpr))) coefficients")

println("\nDerivative accuracy:")
errors_gpr = []
for order in 0:7
    if order == 0
        val = f_gpr(test_point)
    else
        f_deriv = f_gpr
        for i in 1:order
            f_deriv = f_deriv'
        end
        val = f_deriv(test_point)
    end

    truth = ground_truth(order, test_point)
    error = abs(val - truth)
    push!(errors_gpr, error)

    println("  Order $order: value = $(round(val, sigdigits=6)) | error = $(round(error, sigdigits=4))")
end

# Test 2: Least squares (degree 10) for comparison
println("\n" * "-" ^ 80)
println("Test 2: Least squares (degree 10) on noisy data")
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

    println("  Order $order: value = $(round(val, sigdigits=6)) | error = $(round(error, sigdigits=4))")
end

# Test 3: Try different GPR length scales
println("\n" * "-" ^ 80)
println("Test 3: GPR with different length scales")
println("-" ^ 80)

for ls in [0.05, 0.1, 0.2, 0.5]
    println("\nLength scale: $ls")

    se_kernel_test = SE(log(ls), log(1.0))
    gp_test = GP(t_data, y_noisy, MeanZero(), se_kernel_test, log(1e-6))

    gpr_func_test = x -> begin
        μ, σ² = predict_y(gp_test, [x])
        return μ[1]
    end

    try
        f_gpr_test = Fun(gpr_func_test, 0..1)
        n_coeffs = length(coefficients(f_gpr_test))

        # Order 1 error
        e1 = abs(f_gpr_test'(test_point) - ground_truth(1, test_point))

        # Order 7 error
        f7 = f_gpr_test
        for i in 1:7
            f7 = f7'
        end
        e7 = abs(f7(test_point) - ground_truth(7, test_point))

        println("  Coeffs: $n_coeffs | Order 1 err: $(round(e1, sigdigits=4)) | Order 7 err: $(round(e7, sigdigits=4))")
    catch ex
        println("  ERROR: $ex")
    end
end

# Comparison table
println("\n" * "=" ^ 80)
println("COMPARISON")
println("=" ^ 80)

println("\nMethod                          | Coeffs | Order 1 | Order 3 | Order 7")
println("-" ^ 80)
println("GPR (ls=0.1) → ApproxFun        | $(lpad(length(coefficients(f_gpr)), 6)) | $(lpad(round(errors_gpr[2], sigdigits=4), 7)) | $(lpad(round(errors_gpr[4], sigdigits=4), 7)) | $(lpad(round(errors_gpr[8], sigdigits=4), 7))")
println("Least squares deg 10            | $(lpad(length(coefficients(f_ls)), 6)) | $(lpad(round(errors_ls[2], sigdigits=4), 7)) | $(lpad(round(errors_ls[4], sigdigits=4), 7)) | $(lpad(round(errors_ls[8], sigdigits=4), 7))")

println("\n" * "=" ^ 80)
println("RESULT")
println("=" ^ 80)
println("""
Does GPR → ApproxFun work?

GPR advantages:
- Infinitely differentiable (SE kernel)
- No piecewise artifacts
- Already does optimal smoothing
- Truly smooth function

If ApproxFun picks a reasonable number of coefficients for GPR,
this could be the killer workflow!
""")
