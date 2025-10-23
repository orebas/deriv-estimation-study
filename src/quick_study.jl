"""
Streamlined Derivative Estimation Study
========================================
Quick version for generating paper-ready results
"""

using Random
using Statistics
using LinearAlgebra
using Printf
using DataFrames
using CSV

# ODE packages
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using Symbolics

# Approximation methods
using Dierckx
using ForwardDiff

# Plotting
using Plots
using LaTeXStrings

Random.seed!(12345)

# Simplified configuration
const MC_TRIALS = 10
const NOISE_LEVELS = [0.01, 0.05]  # 1%, 5%
const DATA_SIZES = [21, 51]
const MAX_DERIV = 3  # Only up to 3rd derivative

println("=" ^ 70)
println("STREAMLINED DERIVATIVE ESTIMATION STUDY")
println("=" ^ 70)
println("MC trials: $MC_TRIALS, Noise: $NOISE_LEVELS, Sizes: $DATA_SIZES")
println("=" ^ 70)

# Simple Lotka-Volterra system
function lv_system()
    @variables x(t) y(t)
    @parameters α β γ δ

    eqs = [D(x) ~ α * x - β * x * y, D(y) ~ δ * x * y - γ * y]
    @named sys = ODESystem(eqs, t)

    return (
        system = structural_simplify(sys),
        params = [α => 1.5, β => 1.0, γ => 3.0, δ => 1.0],
        ic = [x => 1.0, y => 1.0],
        tspan = (0.0, 10.0),
        obs = [x, y],
        name = "Lotka-Volterra"
    )
end

# Calculate symbolic derivatives
function calc_derivs(sys_def, max_order)
    sys = sys_def.system
    obs = sys_def.obs
    eq_dict = Dict(eq.lhs => eq.rhs for eq in equations(sys))

    derivs = Vector{Vector{Num}}(undef, max_order)
    derivs[1] = [substitute(expand_derivatives(D(o)), eq_dict) for o in obs]

    for ord in 2:max_order
        derivs[ord] = [substitute(expand_derivatives(D(derivs[ord-1][i])), eq_dict)
                      for i in 1:length(obs)]
    end

    return derivs
end

# Generate ground truth
function gen_truth(sys_def, times, max_order)
    derivs = calc_derivs(sys_def, max_order)

    # Create observed equations
    obs_eqs = []
    d_vars = []

    for (i, o) in enumerate(sys_def.obs)
        for ord in 1:max_order
            v = only(@variables $(Symbol("d$(ord)_o$i"))(t))
            push!(d_vars, (i, ord, v))
            push!(obs_eqs, v ~ derivs[ord][i])
        end
    end

    @named ext_sys = ODESystem(equations(sys_def.system), t, observed=obs_eqs)
    prob = ODEProblem(structural_simplify(ext_sys), sys_def.ic, sys_def.tspan, sys_def.params)
    sol = solve(prob, Tsit5(), abstol=1e-10, reltol=1e-10, saveat=times)

    # Extract
    data = Dict(:t => sol.t, :obs => Dict())

    for (i, o) in enumerate(sys_def.obs)
        data[:obs][i] = Dict(0 => sol[o])
        for ord in 1:max_order
            dv = d_vars[findfirst(x -> x[1]==i && x[2]==ord, d_vars)][3]
            data[:obs][i][ord] = sol[dv]
        end
    end

    return data
end

# Methods
function fit_spline(x, y, noise)
    n, m = length(x), mean(abs.(y))
    s = n * (noise * m)^2
    return Spline1D(x, y, k=min(5, length(x)-1), s=s)
end

function fit_finite_diff(x, y)
    return (xi, order) -> begin
        if order == 0
            return y[argmin(abs.(x .- xi))]
        elseif order == 1
            i = argmin(abs.(x .- xi))
            h = x[2] - x[1]
            if i == 1
                return (-3*y[1] + 4*y[2] - y[3]) / (2*h)
            elseif i == length(x)
                return (y[end-2] - 4*y[end-1] + 3*y[end]) / (2*h)
            else
                return (y[i+1] - y[i-1]) / (2*h)
            end
        else
            return NaN
        end
    end
end

# Evaluate methods
function eval_methods(truth, noisy_data, obs_idx, max_order, noise)
    x = truth[:t]
    y = noisy_data[:obs][obs_idx][0]

    results = Dict()

    # Spline
    try
        spl = fit_spline(x, y, noise)
        pred = Dict()
        for ord in 0:max_order
            pred[ord] = [derivative(spl, xi, nu=ord) for xi in x]
        end
        results["Spline"] = pred
    catch
    end

    # Finite difference
    try
        fd = fit_finite_diff(x, y)
        pred = Dict()
        for ord in 0:max_order
            pred[ord] = [fd(xi, ord) for xi in x]
        end
        results["FiniteDiff"] = pred
    catch
    end

    return results
end

# Calculate errors
function calc_errors(pred, true_vals)
    valid = .!isnan.(pred)
    if sum(valid) == 0
        return (rmse=NaN, mae=NaN)
    end

    p, t = pred[valid], true_vals[valid]
    rmse = sqrt(mean((p .- t).^2))
    mae = mean(abs.(p .- t))
    return (rmse=rmse, mae=mae)
end

# Main experiment
println("\nRunning experiments...")
sys = lv_system()
all_results = []

for dsize in DATA_SIZES
    for noise in NOISE_LEVELS
        println("  Size $dsize, Noise $(100*noise)%")

        times = range(sys.tspan[1], sys.tspan[2], length=dsize)
        truth = gen_truth(sys, times, MAX_DERIV)

        for trial in 1:MC_TRIALS
            # Add noise
            rng = MersenneTwister(12345 + trial)
            noisy = deepcopy(truth)
            for (oi, odata) in truth[:obs]
                sig = odata[0]
                nstd = noise * mean(abs.(sig))
                noisy[:obs][oi][0] = sig + nstd * randn(rng, length(sig))
            end

            # Evaluate for first observable
            preds = eval_methods(truth, noisy, 1, MAX_DERIV, noise)

            for (method, pred) in preds
                for ord in 0:MAX_DERIV
                    err = calc_errors(pred[ord], truth[:obs][1][ord])
                    push!(all_results, (
                        data_size=dsize,
                        noise_level=noise,
                        trial=trial,
                        method=method,
                        deriv_order=ord,
                        rmse=err.rmse,
                        mae=err.mae
                    ))
                end
            end
        end
    end
end

# Create DataFrame and summary
df = DataFrame(all_results)
summary = combine(groupby(df, [:method, :deriv_order, :data_size, :noise_level])) do sdf
    (mean_rmse=mean(skipmissing(sdf.rmse)),
     std_rmse=std(skipmissing(sdf.rmse)),
     mean_mae=mean(skipmissing(sdf.mae)))
end

# Save results
mkpath(joinpath(@__DIR__, "..", "results"))
mkpath(joinpath(@__DIR__, "..", "figures"))

CSV.write(joinpath(@__DIR__, "..", "build", "results", "results.csv"), df)
CSV.write(joinpath(@__DIR__, "..", "build", "results", "summary.csv"), summary)

# Generate plots
println("\nGenerating plots...")

# Plot 1: RMSE vs derivative order
for dsize in DATA_SIZES
    for noise in NOISE_LEVELS
        p = plot(title="Data Size=$dsize, Noise=$(100*noise)%",
                xlabel="Derivative Order",
                ylabel="Mean RMSE",
                legend=:topleft,
                yscale=:log10)

        sub = summary[(summary.data_size .== dsize) .& (summary.noise_level .== noise), :]

        for method in unique(sub.method)
            mdata = sub[sub.method .== method, :]
            sort!(mdata, :deriv_order)
            plot!(p, mdata.deriv_order, mdata.mean_rmse,
                 label=method, marker=:circle, linewidth=2)
        end

        savefig(p, joinpath(@__DIR__, "..", "figures",
                "rmse_vs_order_n$(dsize)_noise$(Int(100*noise)).pdf"))
    end
end

# Plot 2: RMSE vs data size
for noise in NOISE_LEVELS
    for ord in 0:2
        p = plot(title="Noise=$(100*noise)%, Derivative Order=$ord",
                xlabel="Number of Data Points",
                ylabel="Mean RMSE",
                legend=:topright,
                yscale=:log10)

        sub = summary[(summary.noise_level .== noise) .& (summary.deriv_order .== ord), :]

        for method in unique(sub.method)
            mdata = sub[sub.method .== method, :]
            sort!(mdata, :data_size)
            plot!(p, mdata.data_size, mdata.mean_rmse,
                 label=method, marker=:circle, linewidth=2)
        end

        savefig(p, joinpath(@__DIR__, "..", "figures",
                "rmse_vs_size_noise$(Int(100*noise))_d$(ord).pdf"))
    end
end

# Generate example trajectory plot
println("\nGenerating example trajectory...")
times = range(sys.tspan[1], sys.tspan[2], length=51)
truth = gen_truth(sys, times, MAX_DERIV)

# Add noise
rng = MersenneTwister(12345)
noisy = deepcopy(truth)
noise = 0.05
sig = truth[:obs][1][0]
noisy[:obs][1][0] = sig + noise * mean(abs.(sig)) * randn(rng, length(sig))

# Fit
spl = fit_spline(times, noisy[:obs][1][0], noise)
spl_pred = [evaluate(spl, t) for t in times]

p = plot(title="Lotka-Volterra: Predator Population",
        xlabel="Time",
        ylabel="Population",
        legend=:topright)
plot!(p, times, truth[:obs][1][0], label="True", linewidth=2)
scatter!(p, times, noisy[:obs][1][0], label="Noisy (5%)", markersize=3, alpha=0.6)
plot!(p, times, spl_pred, label="Spline Fit", linewidth=2, linestyle=:dash)
savefig(p, joinpath(@__DIR__, "..", "figures", "example_fit.pdf"))

# Derivatives plot
p2 = plot(layout=(3,1), size=(800, 900), legend=:topright)
for ord in 1:3
    spl_deriv = [derivative(spl, t, nu=ord) for t in times]
    plot!(p2[ord], times, truth[:obs][1][ord], label="True d$ord", linewidth=2)
    plot!(p2[ord], times, spl_deriv, label="Spline d$ord", linewidth=2,
         linestyle=:dash, xlabel=(ord==3 ? "Time" : ""), ylabel="Derivative $ord")
end
savefig(p2, joinpath(@__DIR__, "..", "figures", "example_derivatives.pdf"))

println("\n" * "=" ^ 70)
println("STUDY COMPLETE")
println("Results: ", joinpath(@__DIR__, "..", "results"))
println("Figures: ", joinpath(@__DIR__, "..", "figures"))
println("=" ^ 70)

# Print summary table
println("\nSummary Statistics:")
println(summary)
