
using ApproxFun
using JSON3
using Statistics

include("methods/julia/rom/approxfun_rom_wrapper.jl")

println("\nTesting ROM WITHOUT densification:")
println("=" ^ 80)

# Load ground truth
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

# Test with different polynomial degrees
for max_degree in [10, 20, 30]
    println("\nDegree $max_degree:")

    try
        result = evaluate_rom("PyNumDiff-SavGol-Tuned-NoDensify", collect(t_eval), 0:7; max_degree=max_degree)

        predictions = result["predictions"]

        for order in 0:7
            if haskey(predictions, order)
                deriv = predictions[order]
                truth = [ground_truth_derivative(order, t) for t in t_eval]
                finite_mask = isfinite.(deriv)

                if sum(finite_mask) > 0
                    errors = abs.(deriv[finite_mask] .- truth[finite_mask])
                    rmse = sqrt(mean(errors.^2))
                    println("  Order $order: RMSE = $(round(rmse, sigdigits=4))")
                else
                    println("  Order $order: All NaN")
                end
            end
        end

    catch e
        println("  ERROR: $e")
    end
end

println("\n" * "=" ^ 80)
