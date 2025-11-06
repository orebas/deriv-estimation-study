"""
Test ApproxFun on Lotka-Volterra: RMSE over entire time series
"""

using ApproxFun
using OrdinaryDiffEq
using Statistics

# Load GP method
include("methods/julia/gp/gaussian_process.jl")
include("methods/julia/common.jl")

println("=" ^ 80)
println("ApproxFun on Lotka-Volterra: RMSE over entire time series")
println("=" ^ 80)

# Generate Lotka-Volterra data
function lotka_volterra!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[1] * u[2]
    du[2] = -γ * u[2] + δ * u[1] * u[2]
end

α, β, γ, δ = 1.5, 1.0, 3.0, 1.0
u0 = [1.0, 1.0]
tspan = (0.0, 10.0)
params = [α, β, γ, δ]

prob = ODEProblem(lotka_volterra!, u0, tspan, params)
sol = solve(prob, Tsit5(), saveat=0.1)

t_data = sol.t
x_true = [u[1] for u in sol.u]

# Add 1% noise
noise_level = 0.01
x_noisy = x_true .+ noise_level .* std(x_true) .* randn(length(x_true))

n = length(t_data)
println("\nData: $n points, noise = $(noise_level * 100)%")

# Compute true derivatives using ForwardDiff on ODE solution
using ForwardDiff

function compute_true_derivatives(sol, t_points, var_idx, max_order)
    true_derivs = Dict{Int, Vector{Float64}}()

    for order in 0:max_order
        true_derivs[order] = zeros(length(t_points))

        for (i, t) in enumerate(t_points)
            if order == 0
                true_derivs[order][i] = sol(t)[var_idx]
            else
                # Nested ForwardDiff
                f = t -> sol(t)[var_idx]
                for j in 1:(order-1)
                    f_prev = f
                    f = t -> ForwardDiff.derivative(f_prev, t)
                end
                true_derivs[order][i] = ForwardDiff.derivative(f, t)
            end
        end
    end

    return true_derivs
end

println("\nComputing true derivatives from ODE solution...")
true_derivs = compute_true_derivatives(sol, t_data, 1, 7)

println("True derivative statistics:")
for order in 0:7
    vals = true_derivs[order]
    println("  Order $order: mean=$(round(mean(vals), sigdigits=4)), std=$(round(std(vals), sigdigits=4)), max_abs=$(round(maximum(abs.(vals)), sigdigits=4))")
end

# =============================================================================
# Test 1: GP-Julia-AD native
# =============================================================================

println("\n" * "-" ^ 80)
println("Test 1: GP-Julia-AD native (TaylorDiff)")
println("-" ^ 80)

result_gp = evaluate_gp_ad(t_data, x_noisy, t_data, collect(0:7))

println("\nRMSE over all time points:")
for order in 0:7
    pred = result_gp.predictions[order]
    truth = true_derivs[order]

    # Compute RMSE only on finite predictions
    finite_mask = isfinite.(pred)
    n_finite = sum(finite_mask)

    if n_finite > 0
        rmse = sqrt(mean((pred[finite_mask] .- truth[finite_mask]).^2))
        println("  Order $order: RMSE = $(round(rmse, sigdigits=6)) ($n_finite/$n finite)")
    else
        println("  Order $order: All NaN")
    end
end

# =============================================================================
# Test 2: GP → ApproxFun
# =============================================================================

println("\n" * "-" ^ 80)
println("Test 2: GP-Julia-AD → ApproxFun (automatic degree)")
println("-" ^ 80)

fitted_gp = fit_gp_ad(t_data, x_noisy)

t_min, t_max = extrema(t_data)
f_approx = Fun(fitted_gp, t_min..t_max)

println("ApproxFun chose $(length(coefficients(f_approx))) coefficients")

println("\nRMSE over all time points:")
for order in 0:7
    pred = zeros(n)

    for (i, t) in enumerate(t_data)
        if order == 0
            pred[i] = f_approx(t)
        else
            f_deriv = f_approx
            for j in 1:order
                f_deriv = f_deriv'
            end
            pred[i] = f_deriv(t)
        end
    end

    truth = true_derivs[order]
    finite_mask = isfinite.(pred)
    n_finite = sum(finite_mask)

    if n_finite > 0
        rmse = sqrt(mean((pred[finite_mask] .- truth[finite_mask]).^2))
        println("  Order $order: RMSE = $(round(rmse, sigdigits=6)) ($n_finite/$n finite)")
    else
        println("  Order $order: All NaN")
    end
end

# =============================================================================
# Test 3: Least squares degree 10
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

println("\nRMSE over all time points:")
for order in 0:7
    pred = zeros(n)

    for (i, t) in enumerate(t_data)
        if order == 0
            pred[i] = f_ls(t)
        else
            f_deriv = f_ls
            for j in 1:order
                f_deriv = f_deriv'
            end
            pred[i] = f_deriv(t)
        end
    end

    truth = true_derivs[order]
    finite_mask = isfinite.(pred)
    n_finite = sum(finite_mask)

    if n_finite > 0
        rmse = sqrt(mean((pred[finite_mask] .- truth[finite_mask]).^2))
        println("  Order $order: RMSE = $(round(rmse, sigdigits=6)) ($n_finite/$n finite)")
    else
        println("  Order $order: All NaN")
    end
end

# =============================================================================
# Test 4: Least squares degree 20
# =============================================================================

println("\n" * "-" ^ 80)
println("Test 4: Least squares (degree 20)")
println("-" ^ 80)

max_degree_20 = 20
basis_20 = [Fun(S, [zeros(k-1); 1.0]) for k in 1:max_degree_20+1]

A_20 = zeros(n, max_degree_20 + 1)
for (i, ti) in enumerate(t_data)
    for (j, φ) in enumerate(basis_20)
        A_20[i, j] = φ(ti)
    end
end

coeffs_20 = A_20 \ x_noisy
f_ls_20 = Fun(S, coeffs_20)

println("Used $(length(coefficients(f_ls_20))) coefficients")

println("\nRMSE over all time points:")
for order in 0:7
    pred = zeros(n)

    for (i, t) in enumerate(t_data)
        if order == 0
            pred[i] = f_ls_20(t)
        else
            f_deriv = f_ls_20
            for j in 1:order
                f_deriv = f_deriv'
            end
            pred[i] = f_deriv(t)
        end
    end

    truth = true_derivs[order]
    finite_mask = isfinite.(pred)
    n_finite = sum(finite_mask)

    if n_finite > 0
        rmse = sqrt(mean((pred[finite_mask] .- truth[finite_mask]).^2))
        println("  Order $order: RMSE = $(round(rmse, sigdigits=6)) ($n_finite/$n finite)")
    else
        println("  Order $order: All NaN")
    end
end

# =============================================================================
# Summary
# =============================================================================

println("\n" * "=" ^ 80)
println("SUMMARY TABLE")
println("=" ^ 80)
println("\nMethod                   | Coeffs | Order 0 | Order 1 | Order 3 | Order 5 | Order 7")
println("-" ^ 95)

# Recompute for table
methods = [
    ("GP-Julia-AD (native)", result_gp.predictions, "N/A"),
]

# Add ApproxFun
pred_approx = Dict{Int, Vector{Float64}}()
for order in 0:7
    pred_approx[order] = zeros(n)
    for (i, t) in enumerate(t_data)
        if order == 0
            pred_approx[order][i] = f_approx(t)
        else
            f_deriv = f_approx
            for j in 1:order
                f_deriv = f_deriv'
            end
            pred_approx[order][i] = f_deriv(t)
        end
    end
end
push!(methods, ("GP → ApproxFun (auto)", pred_approx, "$(length(coefficients(f_approx)))"))

# Add LS deg 10
pred_ls = Dict{Int, Vector{Float64}}()
for order in 0:7
    pred_ls[order] = zeros(n)
    for (i, t) in enumerate(t_data)
        if order == 0
            pred_ls[order][i] = f_ls(t)
        else
            f_deriv = f_ls
            for j in 1:order
                f_deriv = f_deriv'
            end
            pred_ls[order][i] = f_deriv(t)
        end
    end
end
push!(methods, ("Least squares (deg 10)", pred_ls, "11"))

# Add LS deg 20
pred_ls_20 = Dict{Int, Vector{Float64}}()
for order in 0:7
    pred_ls_20[order] = zeros(n)
    for (i, t) in enumerate(t_data)
        if order == 0
            pred_ls_20[order][i] = f_ls_20(t)
        else
            f_deriv = f_ls_20
            for j in 1:order
                f_deriv = f_deriv'
            end
            pred_ls_20[order][i] = f_deriv(t)
        end
    end
end
push!(methods, ("Least squares (deg 20)", pred_ls_20, "21"))

for (name, predictions, coeffs_str) in methods
    rmses = []
    for order in [0, 1, 3, 5, 7]
        pred = predictions[order]
        truth = true_derivs[order]
        finite_mask = isfinite.(pred)
        if sum(finite_mask) > 0
            rmse = sqrt(mean((pred[finite_mask] .- truth[finite_mask]).^2))
            push!(rmses, round(rmse, sigdigits=4))
        else
            push!(rmses, NaN)
        end
    end

    println("$(rpad(name, 24)) | $(lpad(coeffs_str, 6)) | $(lpad(rmses[1], 7)) | $(lpad(rmses[2], 7)) | $(lpad(rmses[3], 7)) | $(lpad(rmses[4], 7)) | $(lpad(rmses[5], 7))")
end

println("\n" * "=" ^ 80)
