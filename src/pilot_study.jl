"""
Pilot Study Driver

Runs a minimal pilot to test the complete pipeline.
"""

include("ground_truth.jl")
include("noise_model.jl")
include("julia_methods.jl")

using Random
using JSON3
using CSV
using DataFrames
using Printf

# Pilot configuration
const PILOT_TRIALS = 2
const PILOT_NOISE_LEVELS = [0.0, 0.01]  # 0%, 1%
const PILOT_DATA_SIZES = [51, 101]
const PILOT_MAX_DERIV = 4  # Start with 0-4 (orders 5-7 are very expensive)
const PYTHON_SCRIPT = joinpath(@__DIR__, "..", "python", "python_methods.py")
const PYTHON_VENV = joinpath(@__DIR__, "..", "python", ".venv", "bin", "python")

println("=" ^ 70)
println("PILOT STUDY: HIGH-ORDER DERIVATIVE ESTIMATION")
println("=" ^ 70)
println("Trials: $PILOT_TRIALS")
println("Noise levels: $PILOT_NOISE_LEVELS")
println("Data sizes: $PILOT_DATA_SIZES")
println("Max derivative order: $PILOT_MAX_DERIV")
println("=" ^ 70)

# Generate ground truth for each data size
println("\nPre-computing ground truth...")
sys = lotka_volterra_system()
ground_truths = Dict()

for dsize in PILOT_DATA_SIZES
    println("  Generating for N=$dsize...")
    times = range(sys.tspan[1], sys.tspan[2], length=dsize)
    truth = generate_ground_truth(sys, times, PILOT_MAX_DERIV)
    ground_truths[dsize] = truth
end

# Run pilot
all_results = []
total_configs = PILOT_TRIALS * length(PILOT_NOISE_LEVELS) * length(PILOT_DATA_SIZES)
config_count = Ref(0)

for dsize in PILOT_DATA_SIZES
    for noise_level in PILOT_NOISE_LEVELS
        for trial in 1:PILOT_TRIALS
            config_count[] += 1
            trial_id = "N$(dsize)_noise$(Int(noise_level*100))_trial$(trial)"

            println("\n[$(config_count[])/$total_configs] Processing $trial_id...")

            # Get ground truth
            truth = ground_truths[dsize]
            times = truth[:t]

            # Add noise
            rng = MersenneTwister(12345 + trial)
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
                    for order in 0:PILOT_MAX_DERIV
                ),
                "config" => Dict(
                    "trial_id" => trial_id,
                    "data_size" => dsize,
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
            # Timeout in seconds (configurable via environment variable)
            timeout_sec = try
                parse(Int, get(ENV, "PYTHON_TIMEOUT", string(300)))
            catch
                300  # Default: 5 minutes
            end

            python_task = @async try
                run(cmd)
                :success
            catch e
                (:error, e)
            end

            # Wait for completion or timeout
            result = timedwait(() -> istaskdone(python_task), timeout_sec; pollint=0.5)

            if result == :timed_out
                @warn "Python script timed out after $(timeout_sec) seconds for $trial_id"
            else
                task_result = fetch(python_task)
                if task_result != :success
                    @warn "Python script failed for $trial_id" exception=task_result[2]
                end
            end

            # === Evaluate Julia methods ===
            println("  Running Julia methods...")
            orders = collect(0:PILOT_MAX_DERIV)
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

                            # Compute RMSE
                            valid = .!isnan.(pred) .& .!isinf.(pred)
                            if sum(valid) > 0
                                rmse = sqrt(mean((pred[valid] .- true_vals[valid]).^2))
                                mae = mean(abs.(pred[valid] .- true_vals[valid]))

                                push!(all_results, (
                                    trial_id = trial_id,
                                    data_size = dsize,
                                    noise_level = noise_level,
                                    trial = trial,
                                    method = result.name,
                                    category = result.category,
                                    language = "Julia",
                                    deriv_order = order,
                                    rmse = rmse,
                                    mae = mae,
                                    timing = result.timing,
                                    valid_points = sum(valid),
                                    total_points = length(pred)
                                ))
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

                                # Compute RMSE
                                valid = .!isnan.(pred) .& .!isinf.(pred)
                                if sum(valid) > 0
                                    rmse = sqrt(mean((pred[valid] .- true_vals[valid]).^2))
                                    mae = mean(abs.(pred[valid] .- true_vals[valid]))

                                    # Determine category
                                    category = if contains(method_name, "GP")
                                        "Gaussian Process"
                                    elseif contains(method_name, "RBF")
                                        "RBF"
                                    elseif contains(method_name, "Spline")
                                        "Spline"
                                    elseif contains(method_name, "FD")
                                        "Finite Difference"
                                    else
                                        "Other"
                                    end

                                    push!(all_results, (
                                        trial_id = trial_id,
                                        data_size = dsize,
                                        noise_level = noise_level,
                                        trial = trial,
                                        method = method_name,
                                        category = category,
                                        language = "Python",
                                        deriv_order = order,
                                        rmse = rmse,
                                        mae = mae,
                                        timing = method_result["timing"],
                                        valid_points = sum(valid),
                                        total_points = length(pred)
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
results_dir = joinpath(@__DIR__, "..", "build", "results", "pilot")
mkpath(results_dir)
CSV.write(joinpath(results_dir, "pilot_results.csv"), df)

# Create summary
println("\nCreating summary...")
summary = combine(groupby(df, [:method, :category, :language, :deriv_order, :noise_level])) do sdf
    (
        mean_rmse = mean(sdf.rmse),
        std_rmse = std(sdf.rmse),
        mean_mae = mean(sdf.mae),
        mean_timing = mean(sdf.timing),
        success_rate = mean(sdf.valid_points .== sdf.total_points)
    )
end

CSV.write(joinpath(results_dir, "pilot_summary.csv"), summary)

# Print top performers
println("\n" * "=" ^ 70)
println("PILOT STUDY COMPLETE")
println("=" ^ 70)

println("\nTop 5 methods by RMSE (noiseless, order 3):")
noiseless_o3 = summary[(summary.noise_level .== 0.0) .& (summary.deriv_order .== 3), :]
sort!(noiseless_o3, :mean_rmse)
for i in 1:min(5, nrow(noiseless_o3))
    row = noiseless_o3[i, :]
    println(@sprintf("  %2d. %-25s RMSE=%.6f (%.3fs)",
                     i, row.method, row.mean_rmse, row.mean_timing))
end

println("\nTop 5 methods by RMSE (1% noise, order 3):")
noisy_o3 = summary[(summary.noise_level .== 0.01) .& (summary.deriv_order .== 3), :]
sort!(noisy_o3, :mean_rmse)
for i in 1:min(5, nrow(noisy_o3))
    row = noisy_o3[i, :]
    println(@sprintf("  %2d. %-25s RMSE=%.6f (%.3fs)",
                     i, row.method, row.mean_rmse, row.mean_timing))
end

println("\nResults saved to: $(results_dir)")
println("=" ^ 70)

