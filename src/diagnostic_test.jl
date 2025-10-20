"""
Diagnostic Test: Check each method for function values and derivatives

Tests each method systematically:
- Order 0: Can it recover function values?
- Order 1: Can it recover first derivatives?
- Order 2: Can it recover second derivatives?
- Order 3: Can it recover third derivatives?
"""

include("ground_truth.jl")

using Printf
using Statistics

println("=" ^ 70)
println("DIAGNOSTIC TEST: Method-by-Method Validation")
println("=" ^ 70)

# Generate test data
println("\nGenerating ground truth...")
sys = lotka_volterra_system()
times = collect(range(sys.tspan[1], sys.tspan[2], length=51))
truth = generate_ground_truth(sys, times, 3)

y_true = truth[:obs][1][0]
y_noisy = y_true  # No noise

# Test orders
test_orders = [0, 1, 2, 3]

# Helper function to compute error metrics
function test_method(predictions, true_vals, method_name, order)
    valid = .!isnan.(predictions) .& .!isinf.(predictions)
    n_valid = sum(valid)

    if n_valid == 0
        return (pass=false, rmse=Inf, mae=Inf, max_err=Inf, n_valid=0)
    end

    errors = abs.(predictions[valid] .- true_vals[valid])
    rmse = sqrt(mean((predictions[valid] .- true_vals[valid]).^2))
    mae = mean(errors)
    max_err = maximum(errors)

    # Determine pass/fail based on order
    threshold = if order == 0
        0.1  # Function values should be very accurate
    elseif order == 1
        1.0  # First derivatives
    elseif order == 2
        10.0  # Second derivatives
    else
        100.0  # Third derivatives
    end

    pass = rmse < threshold

    return (pass=pass, rmse=rmse, mae=mae, max_err=max_err, n_valid=n_valid)
end

println("\n" * "=" ^ 70)
println("TESTING PYTHON METHODS")
println("=" ^ 70)

# Export data for Python
using JSON3
input_json = joinpath(@__DIR__, "..", "data", "input", "diagnostic_test.json")
input_data = Dict(
    "system" => "Lotka-Volterra",
    "observable" => "x(t)",
    "times" => times,
    "y_noisy" => y_noisy,
    "y_true" => y_true,
    "ground_truth_derivatives" => Dict(
        string(order) => truth[:obs][1][order]
        for order in 0:3
    ),
    "config" => Dict(
        "trial_id" => "diagnostic_test",
        "data_size" => 51,
        "noise_level" => 0.0,
        "trial" => 1,
        "tspan" => [sys.tspan[1], sys.tspan[2]]
    )
)

open(input_json, "w") do f
    JSON3.write(f, input_data)
end

# Run Python script with timeout
output_json = joinpath(@__DIR__, "..", "data", "output", "diagnostic_test_results.json")
python_venv = joinpath(@__DIR__, "..", "python", ".venv", "bin", "python")
python_script = joinpath(@__DIR__, "..", "python", "python_methods.py")
cmd = `$python_venv $python_script $input_json $output_json`

println("\nRunning Python methods...")
# Timeout in seconds (configurable via environment variable)
timeout_sec = try
    parse(Int, get(ENV, "PYTHON_TIMEOUT", string(300)))
catch
    300  # Default: 5 minutes
end

python_task = @async try
    run(cmd, devnull, devnull)  # Suppress output
    :success
catch e
    (:error, e)
end

# Wait for completion or timeout
result = timedwait(() -> istaskdone(python_task), timeout_sec; pollint=0.5)

if result == :timed_out
    @warn "Python script timed out after $(timeout_sec) seconds, terminating..."
    exit(1)
else
    task_result = fetch(python_task)
    if task_result == :success
        println("✓ Python methods completed")
    else
        @warn "Python script failed" exception=task_result[2]
        exit(1)
    end
end

# Analyze Python results
if isfile(output_json)
    python_output = JSON3.read(read(output_json, String))

    for (method_name, method_result) in python_output["methods"]
        method_str = string(method_name)
        println("\n" * "-" ^ 70)
        println("Method: $method_str")
        println("-" ^ 70)

        if !method_result["success"]
            println("  ✗ FAILED TO RUN")
            continue
        end

        for order in test_orders
            if haskey(method_result["predictions"], string(order))
                pred = method_result["predictions"][string(order)]
                true_vals = truth[:obs][1][order]

                result = test_method(pred, true_vals, method_str, order)

                status = result.pass ? "✓ PASS" : "✗ FAIL"
                println(@sprintf("  Order %d: %s  RMSE=%.6f  MAE=%.6f  MaxErr=%.6f  Valid=%d/%d",
                                order, status, result.rmse, result.mae, result.max_err,
                                result.n_valid, length(pred)))
            else
                println(@sprintf("  Order %d: ✗ NO DATA", order))
            end
        end
    end
end

println("\n" * "=" ^ 70)
println("TESTING JULIA METHODS")
println("=" ^ 70)

include("julia_methods.jl")

julia_results = evaluate_all_julia_methods(times, y_noisy, times, collect(test_orders);
                                           params=Dict(:noise_level => 0.0))

for result in julia_results
    println("\n" * "-" ^ 70)
    println("Method: $(result.name)")
    println("-" ^ 70)

    if !result.success
        println("  ✗ FAILED TO RUN")
        if !isempty(result.failures)
            for (ord, err) in result.failures
                println("    Order $ord: $err")
            end
        end
        continue
    end

    for order in test_orders
        if haskey(result.predictions, order)
            pred = result.predictions[order]
            true_vals = truth[:obs][1][order]

            test_result = test_method(pred, true_vals, result.name, order)

            status = test_result.pass ? "✓ PASS" : "✗ FAIL"
            println(@sprintf("  Order %d: %s  RMSE=%.6f  MAE=%.6f  MaxErr=%.6f  Valid=%d/%d",
                            order, status, test_result.rmse, test_result.mae, test_result.max_err,
                            test_result.n_valid, length(pred)))
        else
            println(@sprintf("  Order %d: ✗ NO DATA", order))
        end
    end
end

println("\n" * "=" ^ 70)
println("DIAGNOSTIC TEST COMPLETE")
println("=" ^ 70)
