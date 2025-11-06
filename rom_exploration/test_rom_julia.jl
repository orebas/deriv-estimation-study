
using ApproxFun
using JSON3
using Statistics

# Include ROM wrapper
include("methods/julia/rom/approxfun_rom_wrapper.jl")

# Test methods
test_methods = [
    "PyNumDiff-SavGol-Tuned",
    "PyNumDiff-Butter-Tuned"
]

println("\nTesting ROM with ApproxFun:")
println("=" ^ 80)

# Load ground truth for validation
t_eval = range(0, 1, length=101)
omega = 2 * π

function ground_truth_derivative(order, t)
    if order == 0
        return sin(omega * t)
    elseif order == 1
        return omega * cos(omega * t)
    elseif order == 2
        return -(omega^2) * sin(omega * t)
    elseif order == 3
        return -(omega^3) * cos(omega * t)
    elseif order == 4
        return omega^4 * sin(omega * t)
    elseif order == 5
        return omega^5 * cos(omega * t)
    elseif order == 6
        return -(omega^6) * sin(omega * t)
    elseif order == 7
        return -(omega^7) * cos(omega * t)
    end
end

for method_name in test_methods
    println("\n$method_name:")

    try
        # Evaluate ROM
        result = evaluate_rom(method_name, collect(t_eval), 0:7)

        predictions = result["predictions"]

        # Check each order
        for order in 0:7
            if haskey(predictions, order)
                deriv = predictions[order]
                n_finite = sum(isfinite.(deriv))
                n_total = length(deriv)

                if n_finite > 0
                    # Compute RMSE vs ground truth
                    truth = [ground_truth_derivative(order, t) for t in t_eval]
                    finite_mask = isfinite.(deriv)
                    errors = abs.(deriv[finite_mask] .- truth[finite_mask])
                    rmse = sqrt(mean(errors.^2))

                    println("  Order $order: $n_finite/$n_total finite | RMSE: $(round(rmse, sigdigits=4))")
                else
                    println("  Order $order: $n_finite/$n_total finite (all NaN)")
                end
            else
                println("  Order $order: MISSING")
            end
        end

    catch e
        println("  ✗ EXCEPTION: $e")
        println(stacktrace())
    end
end

println("\n" * "=" ^ 80)
println("ROM with ApproxFun test complete")
println("=" ^ 80)
