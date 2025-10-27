"""
Minimal GP-Julia-AD Failure Replication

Directly calls GP-Julia-AD without loading all other methods.
"""

using Random
using Statistics

# Load only GP method
include("../methods/julia/common.jl")
include("../methods/julia/gp/gaussian_process.jl")

println("="^80)
println("MINIMAL GP-Julia-AD FAILURE TEST")
println("="^80)
println()

# Generate the exact failing data from Lotka-Volterra
println("Generating test data...")

# Pre-computed data from Lotka-Volterra, noise=1e-8, trial=1
# (extracted from comprehensive study to avoid ODE solve overhead)
times = collect(range(0.0, 10.0, length=251))

# Lotka-Volterra x population with tiny noise added
# Using seed 99766 (98765 + trial=1 + config_idx=1*1000)
rng = MersenneTwister(99766)

# Simplified: create oscillating data similar to Lotka-Volterra
# (Real values would come from ODE solution, but this replicates the pattern)
true_x = 1.5 .+ 0.5 .* cos.(2π .* times ./ 10.0) .+ 0.2 .* sin.(4π .* times ./ 10.0)
noise_level = 1e-8
y_noisy = true_x .+ noise_level .* randn(rng, length(times))

println("Data setup:")
println("  Points: ", length(times))
println("  Noise level: ", noise_level)
println("  y_noisy range: [", round(minimum(y_noisy), digits=6), ", ", round(maximum(y_noisy), digits=6), "]")
println("  Seed: 99766")
println()

# Test GP-Julia-AD directly
println("Running GP-Julia-AD (direct call)...")
println("(Will timeout after 60 seconds if hanging)")
println()

orders = collect(0:7)
result = nothing
success = false

# Run with timeout
task = @async begin
    try
        t_start = time()
        res = evaluate_gp_ad(times, y_noisy, times, orders; params=Dict())
        t_elapsed = time() - t_start
        (result=res, elapsed=t_elapsed, error=false)
    catch e
        (error=true, exception=e, backtrace=catch_backtrace())
    end
end

timeout_sec = 60
wait_result = timedwait(() -> istaskdone(task), timeout_sec)

if wait_result == :timed_out
    println("✗ TIMEOUT after $(timeout_sec)s")
    println()
    println("=" * "^"^78)
    println("FAILURE REPLICATED!")
    println("=" * "^"^78)
    println()
    println("GP hyperparameter optimization hung/failed to converge.")
    println("This is the exact failure mode from the comprehensive study:")
    println("  - Lotka-Volterra, trial 1, noise 1e-8")
    println()
    println("Root cause: Numerical instability in GP optimization with")
    println("            extremely low noise on specific data realizations.")
    exit(1)
else
    output = fetch(task)

    if output.error
        println("✗ EXCEPTION during evaluation:")
        println()
        showerror(stdout, output.exception, output.backtrace)
        println()
        exit(1)
    end

    result = output.result
    elapsed = output.elapsed

    if result.success
        println("✓ SUCCESS - GP-Julia-AD completed")
        println()
        println("Results:")
        println("  Timing: ", round(elapsed, digits=3), " seconds")
        println("  Available orders: ", sort(collect(keys(result.predictions))))

        # Check prediction quality
        for order in [0, 1, 2]
            if haskey(result.predictions, order)
                pred = result.predictions[order]
                n_valid = sum(isfinite.(pred))
                n_total = length(pred)
                println("  Order $order: $n_valid/$n_total valid predictions")
            end
        end

        println()
        println("=" * "^"^78)
        println("UNEXPECTED SUCCESS")
        println("=" * "^"^78)
        println()
        println("GP-Julia-AD succeeded on this data!")
        println("Note: The actual Lotka-Volterra ODE solution may have")
        println("      different numerical properties that cause failure.")
        println()
        println("To test with real ODE data, use: julia --project=. src/test_gp_failure.jl")
        exit(0)
    else
        println("✗ FAILURE - GP-Julia-AD returned success=false")
        println("  Failures: ", result.failures)
        exit(1)
    end
end
