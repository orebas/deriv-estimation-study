"""
Test GP-Julia-AD on EXACT failing data from comprehensive study

Loads the actual input JSON that caused the failure.
"""

using JSON3

# Load only GP method
include("../methods/julia/common.jl")
include("../methods/julia/gp/gaussian_process.jl")

println("="^80)
println("GP-Julia-AD TEST ON REAL FAILING DATA")
println("="^80)
println()

# Path to the exact failing trial
input_file = "build/data/input/lotka_volterra_noise1e-8_trial1.json"

if !isfile(input_file)
    println("ERROR: Input file not found: $input_file")
    println()
    println("This file should exist from your comprehensive study run.")
    println("If it doesn't exist, the comprehensive study needs to be run first.")
    exit(1)
end

println("Loading data from: $input_file")
data = JSON3.read(read(input_file, String))

times = Vector{Float64}(data["times"])
y_noisy = Vector{Float64}(data["y_noisy"])
orders = collect(0:7)

println()
println("Data loaded:")
println("  Points: ", length(times))
println("  Noise level: ", data["config"]["noise_level"])
println("  Trial: ", data["config"]["trial"])
println("  y_noisy range: [", round(minimum(y_noisy), digits=6), ", ", round(maximum(y_noisy), digits=6), "]")
println()

# Test GP-Julia-AD directly
println("Running GP-Julia-AD on EXACT failing data...")
println("(60 second timeout)")
println()

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
    println("="^80)
    println("FAILURE REPLICATED!")
    println("="^80)
    println()
    println("GP-Julia-AD timed out on the exact data from:")
    println("  - Lotka-Volterra")
    println("  - Noise: 1e-8")
    println("  - Trial: 1")
    println()
    println("This confirms the failure mode: GP hyperparameter optimization")
    println("hangs on this specific data realization.")
else
    output = fetch(task)

    if output.error
        println("✗ EXCEPTION:")
        println()
        showerror(stdout, output.exception, output.backtrace)
        println()
    else
        result = output.result

        if result.success
            println("✓ SUCCESS in ", round(output.elapsed, digits=2), "s")
            println()
            println("Orders completed: ", sort(collect(keys(result.predictions))))
            println()
            println("Note: This is unexpected if the comprehensive study failed.")
            println("      Random seed differences may affect convergence.")
        else
            println("✗ FAILED")
            println("  Failures: ", result.failures)
        end
    end
end
