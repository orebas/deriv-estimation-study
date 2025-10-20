#!/usr/bin/env julia
"""
Background run: Test Matern kernels in comprehensive study
Methods: GP-Julia-Matern-0.5, GP-Julia-Matern-1.5, GP-Julia-Matern-2.5
"""

include("src/ground_truth.jl")
include("src/noise_model.jl")
include("src/julia_methods.jl")

using Random
using JSON3
using CSV
using DataFrames
using Printf
using Statistics

# Study configuration
const NOISE_LEVELS = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 2e-2, 5e-2]
const DATA_SIZE = 101
const MAX_DERIV = 7
const TRIALS_PER_CONFIG = 3
const MATERN_METHODS = ["GP-Julia-Matern-0.5", "GP-Julia-Matern-1.5", "GP-Julia-Matern-2.5"]

println("="^80)
println("MATERN KERNEL COMPREHENSIVE STUDY")
println("="^80)
println("Methods: ", MATERN_METHODS)
println("Noise levels: $NOISE_LEVELS")
println("Trials: $TRIALS_PER_CONFIG")
println("="^80)

# Generate ground truth
sys = lotka_volterra_system()
times = collect(range(sys.tspan[1], sys.tspan[2], length=DATA_SIZE))
truth = generate_ground_truth(sys, times, MAX_DERIV)

# Results storage
all_results = []

for noise_level in NOISE_LEVELS
    for trial in 1:TRIALS_PER_CONFIG
        trial_id = "matern_noise$(noise_level)_trial$(trial)"
        println("\n$trial_id...")

        # Add noise
        rng = MersenneTwister(12345 + trial)
        noisy = add_noise_to_data(truth, noise_level, rng; model=ConstantGaussian)

        x = times
        y_noisy = noisy[:obs][1][0]

        for method_name in MATERN_METHODS
            println("  $method_name...")

            for deriv_order in 0:MAX_DERIV
                t_start = time()

                try
                    result = evaluate_julia_method(method_name, x, y_noisy, x, [deriv_order])
                    elapsed = time() - t_start

                    if result.success && haskey(result.predictions, deriv_order)
                        y_pred = result.predictions[deriv_order]
                        y_true = truth[:obs][1][deriv_order]

                        # Exclude endpoints
                        valid_idx = 2:length(y_true)-1
                        y_pred_clip = y_pred[valid_idx]
                        y_true_clip = y_true[valid_idx]

                        # Compute metrics
                        rmse = sqrt(mean((y_pred_clip .- y_true_clip).^2))
                        mae = mean(abs.(y_pred_clip .- y_true_clip))
                        nrmse = rmse / std(y_true_clip)

                        push!(all_results, Dict(
                            "method" => method_name,
                            "category" => "Gaussian Process",
                            "language" => "Julia",
                            "deriv_order" => deriv_order,
                            "noise_level" => noise_level,
                            "trial" => trial,
                            "rmse" => rmse,
                            "mae" => mae,
                            "nrmse" => nrmse,
                            "timing" => elapsed,
                            "success" => true
                        ))
                    else
                        # Failure case
                        push!(all_results, Dict(
                            "method" => method_name,
                            "category" => "Gaussian Process",
                            "language" => "Julia",
                            "deriv_order" => deriv_order,
                            "noise_level" => noise_level,
                            "trial" => trial,
                            "rmse" => NaN,
                            "mae" => NaN,
                            "nrmse" => NaN,
                            "timing" => elapsed,
                            "success" => false
                        ))
                    end
                catch e
                    println("    Order $deriv_order FAILED: $e")
                    push!(all_results, Dict(
                        "method" => method_name,
                        "category" => "Gaussian Process",
                        "language" => "Julia",
                        "deriv_order" => deriv_order,
                        "noise_level" => noise_level,
                        "trial" => trial,
                        "rmse" => NaN,
                        "mae" => NaN,
                        "nrmse" => NaN,
                        "timing" => NaN,
                        "success" => false
                    ))
                end
            end
        end
    end
end

# Save results
output_dir = joinpath(@__DIR__, "results", "matern")
mkpath(output_dir)

df = DataFrame(all_results)
CSV.write(joinpath(output_dir, "matern_results.csv"), df)

println("\n" * "="^80)
println("✓ MATERN STUDY COMPLETE")
println("Results saved to: results/matern/matern_results.csv")
println("="^80)
