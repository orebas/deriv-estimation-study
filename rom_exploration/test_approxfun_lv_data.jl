"""
Test ApproxFun automatic degree selection on real Lotka-Volterra data with 1% noise

This tests whether GP-Julia-AD → ApproxFun works well on realistic data.
"""

using ApproxFun
using OrdinaryDiffEq

# Load GP method
include("methods/julia/gp/gaussian_process.jl")
include("methods/julia/common.jl")

println("=" ^ 80)
println("ApproxFun on Lotka-Volterra Data (1% noise)")
println("=" ^ 80)

# Generate Lotka-Volterra data
function lotka_volterra!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[1] * u[2]  # Prey
    du[2] = -γ * u[2] + δ * u[1] * u[2] # Predator
end

# Parameters
α, β, γ, δ = 1.5, 1.0, 3.0, 1.0
u0 = [1.0, 1.0]
tspan = (0.0, 10.0)
params = [α, β, γ, δ]

# Solve ODE
prob = ODEProblem(lotka_volterra!, u0, tspan, params)
sol = solve(prob, Tsit5(), saveat=0.1)

t_data = sol.t
x_true = [u[1] for u in sol.u]  # Prey
y_true = [u[2] for u in sol.u]  # Predator

# Add 1% noise
noise_level = 0.01
x_noisy = x_true .+ noise_level .* std(x_true) .* randn(length(x_true))
y_noisy = y_true .+ noise_level .* std(y_true) .* randn(length(y_true))

n = length(t_data)
println("\nData: Lotka-Volterra system")
println("  Time span: $(tspan)")
println("  Points: $n")
println("  Noise level: $(noise_level * 100)% of std")
println("  Prey (x) range: [$(round(minimum(x_true), digits=3)), $(round(maximum(x_true), digits=3))]")
println("  Predator (y) range: [$(round(minimum(y_true), digits=3)), $(round(maximum(y_true), digits=3))]")

# Test point (middle)
test_idx = div(n, 2)
test_point = t_data[test_idx]

println("\nTest point: t = $(round(test_point, digits=3)) (index $test_idx)")

# Compute true derivatives numerically from clean ODE solution
using ForwardDiff

function lv_at_t(t, var_idx)
    # Interpolate solution at time t
    u_t = sol(t)
    return u_t[var_idx]
end

# True derivatives for prey (x)
x_true_derivs = Dict{Int, Float64}()
for order in 0:7
    if order == 0
        x_true_derivs[order] = lv_at_t(test_point, 1)
    else
        # Use nested ForwardDiff
        f = t -> lv_at_t(t, 1)
        for i in 1:(order-1)
            f_prev = f
            f = t -> ForwardDiff.derivative(f_prev, t)
        end
        x_true_derivs[order] = ForwardDiff.derivative(f, test_point)
    end
end

println("\nTrue derivatives of prey (x) at test point:")
for order in 0:7
    println("  Order $order: $(round(x_true_derivs[order], sigdigits=6))")
end

# =============================================================================
# Test 1: GP-Julia-AD native (TaylorDiff)
# =============================================================================

println("\n" * "-" ^ 80)
println("Test 1: GP-Julia-AD native (TaylorDiff)")
println("-" ^ 80)

result_gp = evaluate_gp_ad(t_data, x_noisy, t_data, collect(0:7))

println("GP-Julia-AD derivatives at test point:")
errors_gp = []
for order in 0:7
    pred = result_gp.predictions[order][test_idx]
    truth = x_true_derivs[order]
    error = abs(pred - truth)
    push!(errors_gp, error)

    rel_error = abs(truth) > 1e-10 ? error / abs(truth) : error
    println("  Order $order: value = $(round(pred, sigdigits=6)) | error = $(round(error, sigdigits=4)) | rel = $(round(rel_error, sigdigits=3))")
end

println("\nTiming: $(round(result_gp.timing, digits=3))s")

# =============================================================================
# Test 2: GP-Julia-AD → ApproxFun (automatic degree)
# =============================================================================

println("\n" * "-" ^ 80)
println("Test 2: GP-Julia-AD → ApproxFun (automatic degree)")
println("-" ^ 80)

# Fit GP
fitted_gp = fit_gp_ad(t_data, x_noisy)

# Pass to ApproxFun
println("Building ApproxFun approximation...")
t_min, t_max = extrema(t_data)
f_approx = Fun(fitted_gp, t_min..t_max)

println("ApproxFun chose $(length(coefficients(f_approx))) coefficients")

println("\nDerivatives at test point:")
errors_approx = []
for order in 0:7
    if order == 0
        pred = f_approx(test_point)
    else
        f_deriv = f_approx
        for i in 1:order
            f_deriv = f_deriv'
        end
        pred = f_deriv(test_point)
    end

    truth = x_true_derivs[order]
    error = abs(pred - truth)
    push!(errors_approx, error)

    rel_error = abs(truth) > 1e-10 ? error / abs(truth) : error
    println("  Order $order: value = $(round(pred, sigdigits=6)) | error = $(round(error, sigdigits=4)) | rel = $(round(rel_error, sigdigits=3))")
end

# =============================================================================
# Test 3: Least squares (degree 10)
# =============================================================================

println("\n" * "-" ^ 80)
println("Test 3: Least squares (degree 10)")
println("-" ^ 80)

S = Chebyshev(t_min..t_max)
max_degree = 10
basis = [Fun(S, [zeros(k-1); 1.0]) for k in 1:max_degree+1]

A = zeros(n, max_degree + 1)
for (i, ti) in enumerate(t_data)
    for (j, φ) in enumerate(basis)
        A[i, j] = φ(ti)
    end
end

coeffs = A \ x_noisy
f_ls = Fun(S, coeffs)

println("Used $(length(coefficients(f_ls))) coefficients")

println("\nDerivatives at test point:")
errors_ls = []
for order in 0:7
    if order == 0
        pred = f_ls(test_point)
    else
        f_deriv = f_ls
        for i in 1:order
            f_deriv = f_deriv'
        end
        pred = f_deriv(test_point)
    end

    truth = x_true_derivs[order]
    error = abs(pred - truth)
    push!(errors_ls, error)

    rel_error = abs(truth) > 1e-10 ? error / abs(truth) : error
    println("  Order $order: value = $(round(pred, sigdigits=6)) | error = $(round(error, sigdigits=4)) | rel = $(round(rel_error, sigdigits=3))")
end

# =============================================================================
# Comparison
# =============================================================================

println("\n" * "=" ^ 80)
println("COMPARISON")
println("=" ^ 80)

println("\nMethod                          | Order 1 | Order 3 | Order 5 | Order 7 | Coeffs")
println("-" ^ 90)

println("GP-Julia-AD (TaylorDiff)        | $(lpad(round(errors_gp[2], sigdigits=4), 7)) | $(lpad(round(errors_gp[4], sigdigits=4), 7)) | $(lpad(round(errors_gp[6], sigdigits=4), 7)) | $(lpad(round(errors_gp[8], sigdigits=4), 7)) | N/A")

println("GP → ApproxFun (auto degree)    | $(lpad(round(errors_approx[2], sigdigits=4), 7)) | $(lpad(round(errors_approx[4], sigdigits=4), 7)) | $(lpad(round(errors_approx[6], sigdigits=4), 7)) | $(lpad(round(errors_approx[8], sigdigits=4), 7)) | $(length(coefficients(f_approx)))")

println("Least squares (deg 10)          | $(lpad(round(errors_ls[2], sigdigits=4), 7)) | $(lpad(round(errors_ls[4], sigdigits=4), 7)) | $(lpad(round(errors_ls[6], sigdigits=4), 7)) | $(lpad(round(errors_ls[8], sigdigits=4), 7)) | $(length(coefficients(f_ls)))")

println("\n" * "=" ^ 80)
println("CONCLUSION")
println("=" ^ 80)
println("""
On realistic Lotka-Volterra data with 1% noise:

If GP → ApproxFun uses ~20-30 coefficients:
  → ApproxFun's automatic degree is reasonable
  → Compare accuracy to GP-Julia-AD native and least squares

If GP → ApproxFun uses 1000+ coefficients:
  → ApproxFun is overfitting even on realistic data
  → Manual degree control (least squares) is necessary

Key question: Does ApproxFun's automatic adaptation work better
on realistic ODE data than on simple sine waves?
""")
