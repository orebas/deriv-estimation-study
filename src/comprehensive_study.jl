"""
Comprehensive Derivative Estimation Study

Runs extensive tests across multiple noise levels and derivative orders.
"""

include("ground_truth.jl")
include("noise_model.jl")
include("julia_methods_integrated.jl")

using Random
using JSON3
using CSV
using DataFrames
using Printf

# Study configuration
const NOISE_LEVELS = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 2e-2, 5e-2]  # 0% to 5%
const DATA_SIZE = 101  # Fixed size for consistency
const MAX_DERIV = 7  # Test up to 7th derivative
const TRIALS_PER_CONFIG = 3  # Multiple trials for statistical robustness
const PYTHON_SCRIPT = joinpath(@__DIR__, "..", "python", "python_methods.py")
const PYTHON_VENV = joinpath(@__DIR__, "..", "python", ".venv", "bin", "python")

println("=" ^ 80)
println("COMPREHENSIVE DERIVATIVE ESTIMATION STUDY")
println("=" ^ 80)
println("Noise levels: $NOISE_LEVELS")
println("Data size: $DATA_SIZE")
println("Max derivative order: $MAX_DERIV")
println("Trials per configuration: $TRIALS_PER_CONFIG")
println("=" ^ 80)

# Generate ground truth once
println("\nGenerating ground truth...")
sys = lotka_volterra_system()
times = collect(range(sys.tspan[1], sys.tspan[2], length=DATA_SIZE))
truth = generate_ground_truth(sys, times, MAX_DERIV)

# Run comprehensive study
all_results = []
total_configs = length(NOISE_LEVELS) * TRIALS_PER_CONFIG
config_count = Ref(0)

for noise_level in NOISE_LEVELS
    for trial in 1:TRIALS_PER_CONFIG
        config_count[] += 1
        trial_id = "noise$(Int(noise_level*1e8))e-8_trial$(trial)"

        println("\n[$(config_count[])/$total_configs] Processing noise=$(noise_level), trial=$trial...")

        # Add noise
        rng = MersenneTwister(54321 + trial)  # Changed seed to avoid GP_RBF_Python Trial 1 corruption
        noisy = add_noise_to_data(truth, noise_level, rng; model=ConstantGaussian)

        # Extract observable 1
        y_true = truth[:obs][1][0]
        y_noisy = noisy[:obs][1][0]

        # === Export to JSON for Python ===
        input_json_path = joinpath(@__DIR__, "..", "build", "data", "input", "$(trial_id).json")
        input_data = Dict(
            "system" => "Lotka-Volterra",
            "observable" => "x(t)",
            "times" => times,
            "y_noisy" => y_noisy,
            "y_true" => y_true,
            "ground_truth_derivatives" => Dict(
                string(order) => truth[:obs][1][order]
                for order in 0:MAX_DERIV
            ),
            "config" => Dict(
                "trial_id" => trial_id,
                "data_size" => DATA_SIZE,
                "noise_level" => noise_level,
                "trial" => trial,
                "tspan" => [sys.tspan[1], sys.tspan[2]]
            )
        )

        open(input_json_path, "w") do f
            JSON3.write(f, input_data)
        end

        # === Call Python script with timeout ===
        output_json_path = joinpath(@__DIR__, "..", "build", "data", "output", "$(trial_id)_results.json")
        cmd = `$PYTHON_VENV $PYTHON_SCRIPT $input_json_path $output_json_path`

        println("  Running Python methods...")
        timeout_sec = try
            parse(Int, get(ENV, "PYTHON_TIMEOUT", string(300)))
        catch
            300
        end

        python_task = @async try
            run(cmd)
            :success
        catch e
            (:error, e)
        end

        result = timedwait(() -> istaskdone(python_task), timeout_sec; pollint=0.5)

        if result == :timed_out
            @warn "Python script timed out for $trial_id"
        else
            task_result = fetch(python_task)
            if task_result != :success
                @warn "Python script failed for $trial_id" exception=task_result[2]
            end
        end

        # === Evaluate Julia methods ===
        println("  Running Julia methods...")
        orders = collect(0:MAX_DERIV)
        julia_results = evaluate_all_julia_methods(
            times, y_noisy, times, orders;
            params=Dict(:noise_level => noise_level)
        )

        # === Compute errors ===
        println("  Computing errors...")

        # Process Julia results
        for result in julia_results
            if result.success
                for order in orders
                    if haskey(result.predictions, order)
                        pred = result.predictions[order]
                        true_vals = truth[:obs][1][order]

                        # Compute RMSE excluding endpoints
                        valid = .!isnan.(pred) .& .!isinf.(pred)
                        if sum(valid) > 2
                            # Exclude first and last points
                            idxrng = 2:(length(pred)-1)
                            vmask = valid[idxrng]
                            if sum(vmask) > 0
                                rmse = sqrt(mean((pred[idxrng][vmask] .- true_vals[idxrng][vmask]).^2))
                                mae = mean(abs.(pred[idxrng][vmask] .- true_vals[idxrng][vmask]))

                                # Compute normalized RMSE (nRMSE = RMSE / std(true))
                                true_std = std(true_vals[idxrng][vmask])
                                nrmse = rmse / max(true_std, 1e-12)  # Avoid division by near-zero

                                push!(all_results, (
                                    noise_level = noise_level,
                                    trial = trial,
                                    method = result.name,
                                    category = result.category,
                                    language = "Julia",
                                    deriv_order = order,
                                    rmse = rmse,
                                    mae = mae,
                                    nrmse = nrmse,
                                    timing = result.timing,
                                    valid_points = sum(vmask),
                                    total_points = length(idxrng)
                                ))
                            end
                        end
                    end
                end
            end
        end

        # Process Python results
        if isfile(output_json_path)
            python_output = JSON3.read(read(output_json_path, String))

            for (method_name, method_result) in python_output["methods"]
                if method_result["success"]
                    for order in orders
                        if haskey(method_result["predictions"], string(order))
                            pred = method_result["predictions"][string(order)]
                            true_vals = truth[:obs][1][order]

                            valid = .!isnan.(pred) .& .!isinf.(pred)
                            if sum(valid) > 2
                                # Exclude endpoints
                                idxrng = 2:(length(pred)-1)
                                vmask = valid[idxrng]
                                if sum(vmask) > 0
                                    rmse = sqrt(mean((pred[idxrng][vmask] .- true_vals[idxrng][vmask]).^2))
                                    mae = mean(abs.(pred[idxrng][vmask] .- true_vals[idxrng][vmask]))

                                    # Compute normalized RMSE (nRMSE = RMSE / std(true))
                                    true_std = std(true_vals[idxrng][vmask])
                                    nrmse = rmse / max(true_std, 1e-12)  # Avoid division by near-zero

                                    # Determine category
                                    method_str = string(method_name)
                                    category = if contains(method_str, "GP")
                                        "Gaussian Process"
                                    elseif contains(method_str, "RBF")
                                        "RBF"
                                    elseif contains(method_str, "Spline")
                                        "Spline"
                                    elseif contains(method_str, "FD")
                                        "Finite Difference"
                                    else
                                        "Other"
                                    end

                                    push!(all_results, (
                                        noise_level = noise_level,
                                        trial = trial,
                                        method = method_str,
                                        category = category,
                                        language = "Python",
                                        deriv_order = order,
                                        rmse = rmse,
                                        mae = mae,
                                        nrmse = nrmse,
                                        timing = method_result["timing"],
                                        valid_points = sum(vmask),
                                        total_points = length(idxrng)
                                    ))
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

# Save results
println("\nSaving results...")
df = DataFrame(all_results)
results_dir = joinpath(@__DIR__, "..", "build", "results", "comprehensive")
mkpath(results_dir)
CSV.write(joinpath(results_dir, "comprehensive_results.csv"), df)

# Create summary statistics
println("Creating summary statistics...")
summary = combine(groupby(df, [:method, :category, :language, :deriv_order, :noise_level])) do sdf
    (
        mean_rmse = mean(sdf.rmse),
        std_rmse = std(sdf.rmse),
        min_rmse = minimum(sdf.rmse),
        max_rmse = maximum(sdf.rmse),
        mean_mae = mean(sdf.mae),
        mean_nrmse = mean(sdf.nrmse),
        std_nrmse = std(sdf.nrmse),
        min_nrmse = minimum(sdf.nrmse),
        max_nrmse = maximum(sdf.nrmse),
        mean_timing = mean(sdf.timing),
        trials = nrow(sdf)
    )
end

CSV.write(joinpath(results_dir, "comprehensive_summary.csv"), summary)

println("\n" * "=" ^ 80)
println("COMPREHENSIVE STUDY COMPLETE")
println("=" ^ 80)
println("\nResults saved to: $(results_dir)")
println("  - comprehensive_results.csv (raw data)")
println("  - comprehensive_summary.csv (summary statistics)")
println("=" ^ 80)
