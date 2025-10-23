"""
Test script for Julia methods
"""

include("ground_truth.jl")
include("noise_model.jl")
include("julia_methods_integrated.jl")

using Random
using Printf

function test_julia_methods()
    println("=" ^ 70)
    println("TESTING JULIA METHODS")
    println("=" ^ 70)

    # Generate test data
    println("\nGenerating test data...")
    sys = lotka_volterra_system()
    times = range(sys.tspan[1], sys.tspan[2], length=51)
    truth = generate_ground_truth(sys, times, 7)

    # Add noise
    rng = MersenneTwister(12345)
    noisy = add_noise_to_data(truth, 0.01, rng; model=ConstantGaussian)

    # Extract data
    x = truth[:t]
    y_true = truth[:obs][1][0]
    y_noisy = noisy[:obs][1][0]

    println("\nTesting methods on noisy data (1% noise)...")
    println("Data points: $(length(x))")
    println("Evaluating derivatives 0, 1, 2, 5")

    # Test each method
    orders = [0, 1, 2, 5]

    results = evaluate_all_julia_methods(x, y_noisy, x, orders; params=Dict(:noise_level => 0.01))

    # Print summary
    println("\n" * "=" ^ 70)
    println("RESULTS SUMMARY")
    println("=" ^ 70)

    for result in results
        println("\n$(result.name) ($(result.category)):")
        println("  Success: $(result.success)")
        println("  Timing: $(round(result.timing * 1000, digits=2)) ms")

        if result.success
            for order in orders
                if haskey(result.predictions, order)
                    pred = result.predictions[order]
                    true_vals = truth[:obs][1][order]

                    # Compute error
                    valid = .!isnan.(pred) .& .!isinf.(pred)
                    if sum(valid) > 0
                        rmse = sqrt(mean((pred[valid] .- true_vals[valid]).^2))
                        println("    Order $order: RMSE = $(round(rmse, digits=6)), $(sum(valid))/$(length(pred)) valid")
                    else
                        println("    Order $order: All NaN/Inf")
                    end
                elseif haskey(result.failures, order)
                    println("    Order $order: FAILED - $(result.failures[order])")
                end
            end
        else
            println("  Method failed to fit")
        end
    end

    println("\n✓ Method testing complete!")
    return results
end

# Run test
if abspath(PROGRAM_FILE) == @__FILE__
    test_julia_methods()
end

