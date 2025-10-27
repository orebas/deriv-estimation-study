"""
Replicate GP-Julia-AD Failure

Tests the exact conditions that cause GP-Julia-AD to fail:
- Lotka-Volterra system
- Noise level: 1e-8
- Trial: 1
"""

include("ground_truth.jl")
include("noise_model.jl")
include("julia_methods_integrated.jl")

using Random

println("=" ^ 80)
println("REPLICATING GP-Julia-AD FAILURE")
println("=" ^ 80)
println()

# Test Case 1: Lotka-Volterra, noise 1e-8, trial 1
function test_lotka_volterra_trial1()
    println("Test Case: Lotka-Volterra, noise=1e-8, trial=1")
    println("-" ^ 80)

    # Generate same data as comprehensive study
    sys_def = lotka_volterra_system()
    times = collect(range(sys_def.tspan[1], sys_def.tspan[2], length=251))
    truth = generate_ground_truth(sys_def, times, 7)

    # Use exact same seed calculation
    noise_level = 1e-8
    trial = 1
    config_idx = 1  # lotka_volterra is first in the list
    rng = MersenneTwister(98765 + trial + config_idx * 1000)
    noisy = add_noise_to_data(truth, noise_level, rng; model=ConstantGaussian)

    # Extract observable 1 (x population)
    y_noisy = noisy[:obs][1][0]

    println("Setup:")
    println("  Data points: ", length(times))
    println("  Noise level: ", noise_level)
    println("  y_noisy range: [", round(minimum(y_noisy), digits=6), ", ", round(maximum(y_noisy), digits=6), "]")
    println("  Seed: ", 98765 + trial + config_idx * 1000)
    println()

    # Test GP-Julia-AD with timeout
    println("Running GP-Julia-AD...")
    println("(Will timeout after 60 seconds if hanging)")
    println()

    orders = collect(0:7)
    result = nothing

    # Run in a task with timeout
    task = @async begin
        try
            evaluate_julia_method("GP-Julia-AD", times, y_noisy, times, orders; params=Dict())
        catch e
            (error=true, exception=e, backtrace=catch_backtrace())
        end
    end

    # Wait with timeout
    timeout_sec = 60
    if timedwait(() -> istaskdone(task), timeout_sec) == :timed_out
        println("✗ TIMEOUT after $(timeout_sec)s - GP optimization hung!")
        println()
        println("This is the failure mode observed in comprehensive study.")
        println("The GP hyperparameter optimization fails to converge.")
        return nothing
    else
        result = fetch(task)
    end

    # Check result
    if haskey(result, :error) && result.error
        println("✗ EXCEPTION during evaluation:")
        showerror(stdout, result.exception, result.backtrace)
        println()
        return nothing
    end

    if result.success
        println("✓ SUCCESS - GP-Julia-AD completed")
        println("  Available orders: ", sort(collect(keys(result.predictions))))
        println("  Timing: ", round(result.timing, digits=3), " seconds")

        # Check for NaN/Inf in predictions
        for order in orders
            if haskey(result.predictions, order)
                pred = result.predictions[order]
                n_valid = sum(isfinite.(pred))
                n_total = length(pred)
                if n_valid < n_total
                    println("  ⚠ Order $order: Only $n_valid/$n_total valid predictions")
                end
            end
        end
    else
        println("✗ FAILURE - GP-Julia-AD returned success=false")
        println("  Failures: ", result.failures)
    end

    return result
end

# Test Case 2: Van der Pol, noise 1e-8, trial 8
function test_van_der_pol_trial8()
    println()
    println("=" ^ 80)
    println("Test Case: Van der Pol, noise=1e-8, trial=8")
    println("-" ^ 80)

    sys_def = van_der_pol_system()
    times = collect(range(sys_def.tspan[1], sys_def.tspan[2], length=251))
    truth = generate_ground_truth(sys_def, times, 7)

    noise_level = 1e-8
    trial = 8
    config_idx = 8  # van_der_pol after lotka_volterra (7 noise levels)
    rng = MersenneTwister(98765 + trial + config_idx * 1000)
    noisy = add_noise_to_data(truth, noise_level, rng; model=ConstantGaussian)

    y_noisy = noisy[:obs][1][0]

    println("Setup:")
    println("  Data points: ", length(times))
    println("  Noise level: ", noise_level)
    println("  y_noisy range: [", round(minimum(y_noisy), digits=6), ", ", round(maximum(y_noisy), digits=6), "]")
    println("  Seed: ", 98765 + trial + config_idx * 1000)
    println()

    println("Running GP-Julia-AD...")
    println("(Will timeout after 60 seconds if hanging)")
    println()

    orders = collect(0:7)
    task = @async begin
        try
            evaluate_julia_method("GP-Julia-AD", times, y_noisy, times, orders; params=Dict())
        catch e
            (error=true, exception=e, backtrace=catch_backtrace())
        end
    end

    timeout_sec = 60
    if timedwait(() -> istaskdone(task), timeout_sec) == :timed_out
        println("✗ TIMEOUT after $(timeout_sec)s - GP optimization hung!")
        println()
        return nothing
    else
        result = fetch(task)
    end

    if haskey(result, :error) && result.error
        println("✗ EXCEPTION during evaluation:")
        showerror(stdout, result.exception, result.backtrace)
        println()
        return nothing
    end

    if result.success
        println("✓ SUCCESS")
        println("  Available orders: ", sort(collect(keys(result.predictions))))
        println("  Timing: ", round(result.timing, digits=3), " seconds")
    else
        println("✗ FAILURE")
        println("  Failures: ", result.failures)
    end

    return result
end

# Run tests
println()
result1 = test_lotka_volterra_trial1()

# Uncomment to also test van der pol
# result2 = test_van_der_pol_trial8()

println()
println("=" ^ 80)
println("TEST COMPLETE")
println("=" ^ 80)
println()
println("If you see TIMEOUT or FAILURE above, that's the bug!")
println("These are the exact conditions from the comprehensive study failures.")
println()
