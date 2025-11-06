"""
Debug AAA-ROM Julia wrapper
"""

using JSON3
using BaryRational
using TaylorDiff

println("=" ^ 80)
println("AAA-ROM Debug Test")
println("=" ^ 80)

# Include AAA-ROM wrapper
include("methods/julia/aaa_rom/aaa_rom_wrapper.jl")

# Test loading data
println("\nTest 1: Load densified data")
println("-" ^ 80)

method_name = "PyNumDiff-SavGol-Tuned"

try
    data = load_densified_data(method_name)
    println("✓ Data loaded successfully")
    println("  Keys: ", keys(data))
    println("  method_name: ", get(data, "method_name", "MISSING"))
    println("  n_dense: ", length(get(data, "t_dense", [])))
    println("  n_eval: ", length(get(data, "t_eval", [])))

    # Extract data
    t_dense = Float64.(data["t_dense"])
    y_dense = Float64.(data["y_dense"])

    println("  t_dense range: [", minimum(t_dense), ", ", maximum(t_dense), "]")
    println("  y_dense range: [", minimum(y_dense), ", ", maximum(y_dense), "]")
    println("  All finite? ", all(isfinite.(y_dense)))

catch e
    println("✗ Failed to load data: ", e)
    println(stacktrace())
end

# Test AAA fitting
println("\nTest 2: Build AAA interpolant")
println("-" ^ 80)

try
    data = load_densified_data(method_name)
    t_dense = Float64.(data["t_dense"])
    y_dense = Float64.(data["y_dense"])

    println("Building AAA with $(length(t_dense)) points...")
    aaa_result = BaryRational.aaa(t_dense, y_dense; tol=1e-13, mmax=100, verbose=false)

    println("✓ AAA fit successful")
    println("  Type: ", typeof(aaa_result))
    println("  Has field 'f'? ", hasfield(typeof(aaa_result), :f))

    # Test evaluation
    fitted_func = aaa_result.f
    test_point = 0.5

    println("\nTest evaluation at t=", test_point)
    try
        y_eval = fitted_func(test_point)
        println("  ✓ f(0.5) = ", y_eval)
    catch e
        println("  ✗ Evaluation failed: ", e)
    end

    # Test derivative
    println("\nTest derivative at t=", test_point)
    try
        dy_eval = TaylorDiff.derivative(fitted_func, test_point, Val(1))
        println("  ✓ f'(0.5) = ", dy_eval)
    catch e
        println("  ✗ Derivative failed: ", e)
    end

catch e
    println("✗ AAA fitting failed: ", e)
    println(stacktrace())
end

# Test full evaluate_aaa_rom
println("\nTest 3: Full evaluate_aaa_rom")
println("-" ^ 80)

try
    t_eval = range(0, 1, length=101)
    result = evaluate_aaa_rom(method_name, collect(t_eval), 0:7; aaa_tol=1e-13)

    println("✓ evaluate_aaa_rom completed")
    println("  Keys: ", keys(result))
    println("  Has predictions? ", haskey(result, "predictions"))
    println("  Has failures? ", haskey(result, "failures"))

    if haskey(result, "predictions")
        predictions = result["predictions"]
        println("\n  Predictions:")
        for order in 0:7
            if haskey(predictions, order)
                deriv = predictions[order]
                n_finite = sum(isfinite.(deriv))
                if n_finite > 0
                    println("    Order $order: $n_finite/$(length(deriv)) finite, range [$(minimum(deriv[isfinite.(deriv)])), $(maximum(deriv[isfinite.(deriv)]))]")
                else
                    println("    Order $order: $n_finite/$(length(deriv)) finite (all NaN)")
                    # Print first few values for debugging
                    println("      First 5 values: ", deriv[1:min(5, length(deriv))])
                end
            else
                println("    Order $order: MISSING from predictions")
            end
        end
    end

    if haskey(result, "failures")
        failures = result["failures"]
        if !isempty(failures)
            println("\n  Failures:")
            for (order, msg) in failures
                println("    Order $order: ", msg)
            end
        end
    end

catch e
    println("✗ evaluate_aaa_rom failed: ", e)
    println(stacktrace())
end

println("\n" * "=" ^ 80)
println("Debug test complete")
println("=" ^ 80)
