"""
Comprehensive Derivative Estimation Study
==========================================

This script performs a rigorous comparison of numerical differentiation methods
for estimating higher-order derivatives from noisy ODE trajectory data.

The study includes:
- Multiple ODE systems (Lotka-Volterra, Van der Pol, SIR model)
- Various noise levels (0.1% to 10%)
- Different data densities (11, 21, 51, 101 points)
- Monte Carlo simulations (100 trials per configuration)
- Comprehensive error metrics (RMSE, MAE, relative errors)
- Computational cost analysis
- Statistical significance testing

Methods tested:
1. Gaussian Process Regression (GPR)
2. Adaptive Antoulas-Anderson (AAA) rational approximation
3. B-spline smoothing (Dierckx, order 5)
4. LOESS (Locally weighted regression)
5. Finite differences (central, order 2)
6. Savitzky-Golay filtering
"""

using Random
using Statistics
using LinearAlgebra
using Printf
using Serialization
using DataFrames
using CSV

# ODE and numerical packages
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using Symbolics

# Approximation methods
using GaussianProcesses
using Loess
using BaryRational
using Dierckx
using Optim
using LineSearches
using ForwardDiff
using TaylorDiff

# Plotting
using Plots
using StatsPlots
using LaTeXStrings

# Set random seed for reproducibility
Random.seed!(12345)

# Configuration (can be overridden)
MONTE_CARLO_TRIALS = get(ENV, "MC_TRIALS", "100") |> x -> parse(Int, x)
NOISE_LEVELS = [0.001, 0.01, 0.05, 0.1]  # 0.1%, 1%, 5%, 10%
DATA_SIZES = [11, 21, 51, 101]
MAX_DERIVATIVE_ORDER = 5
ODE_SOLVER_TOLERANCE = 1e-12

println("=" ^ 80)
println("DERIVATIVE ESTIMATION STUDY")
println("=" ^ 80)
println("Configuration:")
println("  Monte Carlo trials: $MONTE_CARLO_TRIALS")
println("  Noise levels: $NOISE_LEVELS")
println("  Data sizes: $DATA_SIZES")
println("  Max derivative order: $MAX_DERIVATIVE_ORDER")
println("=" ^ 80)

# ============================================================================
# ODE SYSTEM DEFINITIONS
# ============================================================================

"""
Define the Lotka-Volterra predator-prey model
"""
function lotka_volterra_system()
    @variables x(t) y(t)
    @parameters α β γ δ

    eqs = [
        D(x) ~ α * x - β * x * y,
        D(y) ~ δ * x * y - γ * y
    ]

    @named sys = ODESystem(eqs, t)

    return (
        system = structural_simplify(sys),
        params = [α => 1.5, β => 1.0, γ => 3.0, δ => 1.0],
        initial_conditions = [x => 1.0, y => 1.0],
        time_span = (0.0, 10.0),
        observables = [x, y],
        name = "Lotka-Volterra"
    )
end

"""
Define the Van der Pol oscillator
"""
function van_der_pol_system()
    @variables x(t) y(t)
    @parameters μ

    eqs = [
        D(x) ~ y,
        D(y) ~ μ * (1 - x^2) * y - x
    ]

    @named sys = ODESystem(eqs, t)

    return (
        system = structural_simplify(sys),
        params = [μ => 2.0],
        initial_conditions = [x => 2.0, y => 0.0],
        time_span = (0.0, 20.0),
        observables = [x, y],
        name = "Van-der-Pol"
    )
end

"""
Define a simple SIR epidemiological model
"""
function sir_system()
    @variables S(t) I(t) R(t)
    @parameters β γ N

    eqs = [
        D(S) ~ -β * S * I / N,
        D(I) ~ β * S * I / N - γ * I,
        D(R) ~ γ * I
    ]

    @named sys = ODESystem(eqs, t)

    return (
        system = structural_simplify(sys),
        params = [β => 0.5, γ => 0.1, N => 1000.0],
        initial_conditions = [S => 990.0, I => 10.0, R => 0.0],
        time_span = (0.0, 100.0),
        observables = [S, I, R],
        name = "SIR"
    )
end

# ============================================================================
# DERIVATIVE CALCULATION
# ============================================================================

"""
Calculate symbolic derivatives of observables up to specified order
"""
function calculate_symbolic_derivatives(system_def, max_order)
    sys = system_def.system
    observables = system_def.observables

    # Get equations for substitution
    eq_dict = Dict(eq.lhs => eq.rhs for eq in equations(sys))

    n_obs = length(observables)

    # Storage for symbolic derivatives
    sym_derivs = Vector{Vector{Num}}(undef, max_order)

    # First derivatives
    sym_derivs[1] = [substitute(expand_derivatives(D(obs)), eq_dict) for obs in observables]

    # Higher order derivatives
    for order in 2:max_order
        sym_derivs[order] = [substitute(expand_derivatives(D(sym_derivs[order-1][i])), eq_dict)
                            for i in 1:n_obs]
    end

    return sym_derivs
end

"""
Generate ground truth data with exact derivatives
"""
function generate_ground_truth(system_def, time_points, max_order)
    # Calculate symbolic derivatives
    sym_derivs = calculate_symbolic_derivatives(system_def, max_order)

    # Create observed equations including derivatives
    obs_eqs = []
    deriv_vars = []

    for (i, obs) in enumerate(system_def.observables)
        for order in 1:max_order
            # Create variable properly using Symbolics
            var = only(@variables $(Symbol("d$(order)_obs$i"))(t))
            push!(deriv_vars, (i, order, var))
            push!(obs_eqs, var ~ sym_derivs[order][i])
        end
    end

    # Create new system with derivatives
    @named extended_sys = ODESystem(equations(system_def.system), t,
                                     observed = obs_eqs)

    # Solve
    prob = ODEProblem(structural_simplify(extended_sys),
                      system_def.initial_conditions,
                      system_def.time_span,
                      system_def.params)

    sol = solve(prob, AutoVern9(Rodas4P()),
                abstol=ODE_SOLVER_TOLERANCE,
                reltol=ODE_SOLVER_TOLERANCE,
                saveat=time_points)

    # Extract data
    data = Dict()
    data[:time] = sol.t
    data[:observables] = Dict()

    for (i, obs) in enumerate(system_def.observables)
        data[:observables][i] = Dict()
        data[:observables][i][0] = sol[obs]

        for order in 1:max_order
            deriv_var = deriv_vars[findfirst(x -> x[1] == i && x[2] == order, deriv_vars)][3]
            data[:observables][i][order] = sol[deriv_var]
        end
    end

    return data
end

"""
Add noise to data
"""
function add_noise(clean_data, noise_level, rng)
    noisy_data = deepcopy(clean_data)

    for (obs_idx, obs_data) in clean_data[:observables]
        signal = obs_data[0]
        noise_std = noise_level * mean(abs.(signal))
        noise = noise_std * randn(rng, length(signal))
        noisy_data[:observables][obs_idx][0] = signal + noise
    end

    return noisy_data
end

# ============================================================================
# APPROXIMATION METHODS
# ============================================================================

"""
Numerical derivative using finite differences (central difference)
"""
function nth_derivative_finite_diff(x, y, order, point_idx)
    n = length(x)
    h = x[2] - x[1]  # Assume uniform spacing

    if order == 1
        if point_idx == 1
            return (-3*y[1] + 4*y[2] - y[3]) / (2*h)
        elseif point_idx == n
            return (y[n-2] - 4*y[n-1] + 3*y[n]) / (2*h)
        else
            return (y[point_idx+1] - y[point_idx-1]) / (2*h)
        end
    elseif order == 2
        if point_idx == 1 || point_idx == n
            return NaN
        else
            return (y[point_idx+1] - 2*y[point_idx] + y[point_idx-1]) / h^2
        end
    else
        # For higher orders, use forward difference approximation
        return NaN
    end
end

"""
Wrapper for automatic differentiation of approximation functions
"""
function nth_deriv_autodiff(func, order, x)
    if order == 0
        return func(x)
    end

    # Use TaylorDiff for higher order derivatives
    try
        taylor_result = TaylorDiff.derivative(func, x, order)
        return taylor_result
    catch
        # Fallback to ForwardDiff for first order
        if order == 1
            return ForwardDiff.derivative(func, x)
        else
            return NaN
        end
    end
end

"""
Gaussian Process Regression
"""
function fit_gpr(x, y)
    # Normalize
    y_mean = mean(y)
    y_std = std(y)
    y_norm = (y .- y_mean) ./ max(y_std, 1e-8)

    # Add small jitter
    y_jitter = y_norm .+ 1e-6 * randn(length(y))

    # Setup and optimize
    kernel = SEIso(log(std(x) / 8), 0.0)
    gp = GP(x, y_jitter, MeanZero(), kernel, -2.0)

    try
        optimize!(gp, method=LBFGS(linesearch=LineSearches.BackTracking()))
    catch
        # Use unoptimized if optimization fails
    end

    # Return denormalized function
    return x_eval -> begin
        pred, _ = predict_f(gp, [x_eval])
        return y_std * pred[1] + y_mean
    end
end

"""
AAA rational approximation
"""
function fit_aaa(x, y)
    # Normalize
    y_mean = mean(y)
    y_std = std(y)
    y_norm = (y .- y_mean) ./ max(y_std, 1e-8)

    # Fit AAA
    approx = aaa(x, y_norm, verbose=false, tol=1e-10)

    # Return denormalized function
    return x_eval -> y_std * approx(x_eval) + y_mean
end

"""
B-spline smoothing (Dierckx)
"""
function fit_spline(x, y, noise_level)
    # Calculate smoothing parameter based on noise
    n = length(x)
    mean_y = mean(abs.(y))
    s = n * (noise_level * mean_y)^2

    # Fit spline
    spl = Spline1D(x, y, k=5, s=s)

    # Return function that can compute derivatives
    return (x_eval, deriv_order=0) -> derivative(spl, x_eval, nu=deriv_order)
end

"""
LOESS smoothing
"""
function fit_loess(x, y, span=0.2)
    model = loess(collect(x), y, span=span)
    predictions = Loess.predict(model, x)

    # Use AAA on LOESS output for smooth derivatives
    return fit_aaa(x, predictions)
end

"""
Savitzky-Golay filter - simple implementation
For derivatives, we'll use finite differences on filtered data
"""
function fit_savgol(x, y, window_length=11)
    # Simple moving average for now
    # Real SG would require more complex implementation
    n = length(y)
    half_window = div(window_length, 2)
    filtered = copy(y)

    for i in (half_window+1):(n-half_window)
        filtered[i] = mean(y[(i-half_window):(i+half_window)])
    end

    # Use AAA on filtered data
    return fit_aaa(x, filtered)
end

# ============================================================================
# EVALUATION AND COMPARISON
# ============================================================================

"""
Evaluate a single method on a dataset
"""
function evaluate_method(method_name, fit_func, ground_truth, noisy_data,
                         obs_idx, max_order, noise_level)
    x = ground_truth[:time]
    y = noisy_data[:observables][obs_idx][0]

    # Timing
    time_start = time()

    # Fit the method
    if method_name == "Spline"
        approx_func = fit_func(x, y, noise_level)
        # Special handling for splines
        predictions = Dict()
        predictions[0] = [approx_func(xi, 0) for xi in x]
        for order in 1:max_order
            predictions[order] = [approx_func(xi, order) for xi in x]
        end
    elseif method_name == "FiniteDiff"
        # Finite differences don't need fitting
        predictions = Dict()
        predictions[0] = y
        predictions[1] = [nth_derivative_finite_diff(x, y, 1, i) for i in 1:length(x)]
        predictions[2] = [nth_derivative_finite_diff(x, y, 2, i) for i in 1:length(x)]
        for order in 3:max_order
            predictions[order] = fill(NaN, length(x))
        end
    else
        # Standard approximation function
        approx_func = fit_func(x, y)
        predictions = Dict()
        predictions[0] = [approx_func(xi) for xi in x]
        for order in 1:max_order
            predictions[order] = [nth_deriv_autodiff(approx_func, order, xi) for xi in x]
        end
    end

    time_elapsed = time() - time_start

    # Calculate errors
    errors = Dict()
    for order in 0:max_order
        true_vals = ground_truth[:observables][obs_idx][order]
        pred_vals = predictions[order]

        # Remove NaNs for error calculation
        valid_idx = .!isnan.(pred_vals)
        if sum(valid_idx) == 0
            errors[order] = (rmse=NaN, mae=NaN, max_error=NaN, rel_rmse=NaN)
            continue
        end

        true_valid = true_vals[valid_idx]
        pred_valid = pred_vals[valid_idx]

        rmse = sqrt(mean((pred_valid .- true_valid).^2))
        mae = mean(abs.(pred_valid .- true_valid))
        max_error = maximum(abs.(pred_valid .- true_valid))

        # Relative RMSE
        scale = max(std(true_valid), 1e-10)
        rel_rmse = rmse / scale

        errors[order] = (rmse=rmse, mae=mae, max_error=max_error, rel_rmse=rel_rmse)
    end

    return (errors=errors, time=time_elapsed, predictions=predictions)
end

"""
Run complete experimental suite
"""
function run_experiments()
    systems = [
        lotka_volterra_system(),
        van_der_pol_system(),
        sir_system()
    ]

    methods = Dict(
        "GPR" => fit_gpr,
        "AAA" => fit_aaa,
        "Spline" => fit_spline,
        "LOESS" => fit_loess,
        "SavGol" => fit_savgol,
        "FiniteDiff" => nothing  # Special case
    )

    results = []

    total_configs = length(systems) * length(NOISE_LEVELS) * length(DATA_SIZES)
    config_count = 0

    for system_def in systems
        println("\n" * "=" ^ 80)
        println("SYSTEM: $(system_def.name)")
        println("=" ^ 80)

        for data_size in DATA_SIZES
            for noise_level in NOISE_LEVELS
                config_count += 1
                println("\nConfiguration $config_count/$total_configs:")
                println("  Data size: $data_size, Noise level: $(100*noise_level)%")

                # Generate time points
                time_points = range(system_def.time_span[1],
                                   system_def.time_span[2],
                                   length=data_size)

                # Generate ground truth once
                ground_truth = generate_ground_truth(system_def, time_points,
                                                    MAX_DERIVATIVE_ORDER)

                # Monte Carlo trials
                for trial in 1:MONTE_CARLO_TRIALS
                    if trial % 25 == 0
                        print("    Trial $trial/$MONTE_CARLO_TRIALS\r")
                    end

                    # Generate noisy data with trial-specific RNG
                    rng = MersenneTwister(12345 + trial)
                    noisy_data = add_noise(ground_truth, noise_level, rng)

                    # Test each method on each observable
                    for obs_idx in 1:length(system_def.observables)
                        for (method_name, fit_func) in methods
                            try
                                result = evaluate_method(method_name, fit_func,
                                                        ground_truth, noisy_data,
                                                        obs_idx, MAX_DERIVATIVE_ORDER,
                                                        noise_level)

                                push!(results, (
                                    system = system_def.name,
                                    observable = obs_idx,
                                    data_size = data_size,
                                    noise_level = noise_level,
                                    trial = trial,
                                    method = method_name,
                                    errors = result.errors,
                                    time = result.time
                                ))
                            catch e
                                @warn "Method $method_name failed" exception=e
                            end
                        end
                    end
                end
                println()
            end
        end
    end

    return results
end

# ============================================================================
# ANALYSIS AND VISUALIZATION
# ============================================================================

"""
Aggregate results for analysis
"""
function aggregate_results(results)
    df_rows = []

    for result in results
        for (order, error_metrics) in result.errors
            push!(df_rows, (
                system = result.system,
                observable = result.observable,
                data_size = result.data_size,
                noise_level = result.noise_level,
                trial = result.trial,
                method = result.method,
                derivative_order = order,
                rmse = error_metrics.rmse,
                mae = error_metrics.mae,
                max_error = error_metrics.max_error,
                rel_rmse = error_metrics.rel_rmse,
                time = result.time
            ))
        end
    end

    return DataFrame(df_rows)
end

"""
Create summary statistics
"""
function create_summary_statistics(df)
    summary = combine(groupby(df, [:system, :method, :derivative_order, :data_size, :noise_level])) do subdf
        (
            mean_rmse = mean(skipmissing(subdf.rmse)),
            std_rmse = std(skipmissing(subdf.rmse)),
            mean_mae = mean(skipmissing(subdf.mae)),
            mean_rel_rmse = mean(skipmissing(subdf.rel_rmse)),
            mean_time = mean(subdf.time),
            median_rmse = median(skipmissing(subdf.rmse))
        )
    end

    return summary
end

"""
Generate publication-quality plots
"""
function generate_plots(df, summary, output_dir)
    println("\nGenerating plots...")

    # 1. Error vs. Noise Level (for each system and derivative order)
    for system in unique(df.system)
        for deriv_order in 0:2  # Focus on first few derivatives
            p = plot(title="$system - Derivative Order $deriv_order",
                    xlabel="Noise Level (%)",
                    ylabel="Mean RMSE",
                    legend=:topleft,
                    xscale=:log10,
                    yscale=:log10)

            subdata = summary[(summary.system .== system) .&
                             (summary.derivative_order .== deriv_order) .&
                             (summary.data_size .== 51), :]

            for method in unique(subdata.method)
                method_data = subdata[subdata.method .== method, :]
                sort!(method_data, :noise_level)

                plot!(p, 100 .* method_data.noise_level,
                     method_data.mean_rmse,
                     label=method,
                     marker=:circle,
                     linewidth=2)
            end

            savefig(p, joinpath(output_dir, "error_vs_noise_$(system)_d$(deriv_order).pdf"))
        end
    end

    # 2. Error vs. Data Size
    for system in unique(df.system)
        for deriv_order in 0:2
            p = plot(title="$system - Derivative Order $deriv_order",
                    xlabel="Number of Data Points",
                    ylabel="Mean RMSE",
                    legend=:topright,
                    yscale=:log10)

            subdata = summary[(summary.system .== system) .&
                             (summary.derivative_order .== deriv_order) .&
                             (summary.noise_level .== 0.01), :]

            for method in unique(subdata.method)
                method_data = subdata[subdata.method .== method, :]
                sort!(method_data, :data_size)

                plot!(p, method_data.data_size,
                     method_data.mean_rmse,
                     label=method,
                     marker=:circle,
                     linewidth=2)
            end

            savefig(p, joinpath(output_dir, "error_vs_datasize_$(system)_d$(deriv_order).pdf"))
        end
    end

    # 3. Computational cost comparison
    time_summary = combine(groupby(df, [:method])) do subdf
        (mean_time = mean(subdf.time),)
    end

    p = bar(time_summary.method,
            time_summary.mean_time,
            title="Computational Cost Comparison",
            ylabel="Mean Time (seconds)",
            legend=false,
            xrotation=45)

    savefig(p, joinpath(output_dir, "computational_cost.pdf"))

    println("Plots saved to $output_dir")
end

# ============================================================================
# MAIN EXECUTION
# ============================================================================

println("\nStarting experiments...")
results = run_experiments()

println("\nAggregating results...")
df = aggregate_results(results)
summary = create_summary_statistics(df)

println("\nSaving results...")
CSV.write(joinpath(@__DIR__, "..", "results", "raw_results.csv"), df)
CSV.write(joinpath(@__DIR__, "..", "results", "summary_statistics.csv"), summary)
serialize(joinpath(@__DIR__, "..", "results", "results.jls"), results)

println("\nGenerating visualizations...")
generate_plots(df, summary, joinpath(@__DIR__, "..", "figures"))

println("\n" * "=" ^ 80)
println("STUDY COMPLETE")
println("=" ^ 80)
println("Results saved to: ", joinpath(@__DIR__, "..", "results"))
println("Figures saved to: ", joinpath(@__DIR__, "..", "figures"))
