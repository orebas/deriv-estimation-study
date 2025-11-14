"""
Test script to reproduce GP-TaylorAD-Julia failures

This script replicates the exact conditions of failing GP trials to help debug
and add robustness improvements.

Usage:
    julia --project=. src/test_gp_failure.jl
"""

include("ground_truth.jl")
include("noise_model.jl")
include("julia_methods_integrated.jl")
include("config_loader.jl")

using Random
using Printf

# Known failing configurations from comprehensive study
FAILING_CONFIGS = [
    # (ode_key, noise_level, trial)
    ("lotka_volterra", 1.0e-8, 1),
    ("lotka_volterra", 1.0e-8, 5),
    ("lotka_volterra", 0.01, 6),
    ("van_der_pol", 0.001, 1),
    ("lorenz", 1.0e-8, 1),
    ("lorenz", 0.0001, 5),
    ("lorenz", 0.001, 3),
    ("lorenz", 0.001, 6),
    ("lorenz", 0.01, 5),
]

println("="^80)
println("GP-TaylorAD-Julia Failure Reproduction Test")
println("="^80)
println()

# Load configuration
const CONFIG = get_comprehensive_config()
const DATA_SIZE = CONFIG.data_size
const MAX_DERIV = CONFIG.max_derivative_order

# Load ODE systems
enabled_ode_keys = get_enabled_ode_systems()
ode_systems = get_all_ode_systems(enabled_ode_keys)

println("Testing $(length(FAILING_CONFIGS)) known failing configurations...")
println()

success_count = 0
failure_count = 0
failure_details = []

for (ode_key, noise_level, trial) in FAILING_CONFIGS
    trial_id = "$(ode_key)_noise$(Int(noise_level*1e8))e-8_trial$(trial)"

    print("Testing $trial_id... ")
    flush(stdout)

    # Generate ground truth for this ODE system
    sys_def = ode_systems[ode_key]
    times = collect(range(sys_def.tspan[1], sys_def.tspan[2], length = DATA_SIZE))
    truth = generate_ground_truth(sys_def, times, MAX_DERIV)

    # Add noise with same seed as comprehensive study
    config_idx = 1  # Simplified - in real study this would be calculated
    seed = 98765 + trial + config_idx * 1000
    rng = MersenneTwister(seed)
    noisy = add_noise_to_data(truth, noise_level, rng; model = ConstantGaussian)

    # Extract observable 1
    y_true = truth[:obs][1][0]
    y_noisy = noisy[:obs][1][0]

    # Test GP-TaylorAD-Julia specifically
    orders = collect(0:MAX_DERIV)

    try
        result = evaluate_gp_ad(times, y_noisy, times, orders; params=Dict())

        if result.success
            println("✓ SUCCESS")
            success_count += 1
        else
            println("✗ FAILED (returned success=false)")
            failure_count += 1
            push!(failure_details, (
                trial_id = trial_id,
                error = result.failures,
                data_stats = (
                    n_points = length(times),
                    y_range = (minimum(y_noisy), maximum(y_noisy)),
                    y_std = std(y_noisy),
                    noise_level = noise_level
                )
            ))
        end
    catch e
        println("✗ EXCEPTION")
        failure_count += 1
        error_msg = sprint(showerror, e, catch_backtrace())
        push!(failure_details, (
            trial_id = trial_id,
            error = error_msg,
            exception_type = typeof(e),
            data_stats = (
                n_points = length(times),
                y_range = (minimum(y_noisy), maximum(y_noisy)),
                y_std = std(y_noisy),
                noise_level = noise_level
            )
        ))
    end
end

println()
println("="^80)
println("SUMMARY")
println("="^80)
println("Total tests: $(length(FAILING_CONFIGS))")
println("Successes: $success_count")
println("Failures: $failure_count")
println()

if failure_count > 0
    println("="^80)
    println("FAILURE DETAILS")
    println("="^80)
    println()

    for (i, detail) in enumerate(failure_details)
        println("[$i] $(detail.trial_id)")
        println("    Error: $(detail.error)")
        if haskey(detail, :exception_type)
            println("    Exception type: $(detail.exception_type)")
        end
        println("    Data stats:")
        println("      Points: $(detail.data_stats.n_points)")
        println("      Y range: $(detail.data_stats.y_range)")
        println("      Y std: $(detail.data_stats.y_std)")
        println("      Noise level: $(detail.data_stats.noise_level)")
        println()
    end
end

println("="^80)
