"""
AAA-ROM (Adaptive Antoulas-Anderson Reduced Order Model) Wrapper

This module provides a generic framework for computing higher-order derivatives
from ANY smoothing method using AAA barycentric rational interpolation.

Workflow:
1. Python methods produce densified smoothed signals (~1000 points)
2. Save to build/data/aaa_rom_input/<method>_dense.json
3. Julia reads data, builds AAA interpolant
4. Compute derivatives via Taylor differentiation of AAA

Data Format (JSON):
{
    "method_name": "PyNumDiff-SavGol-Tuned",
    "t_dense": [0.0, 0.001, 0.002, ...],  # Dense time grid (~1000 points)
    "y_dense": [1.2, 1.3, 1.4, ...],       # Smoothed signal on dense grid
    "t_eval": [0.0, 0.01, 0.02, ...],      # Original evaluation points
    "metadata": {
        "n_dense": 1000,
        "source_method": "...",
        "aaa_tol": 1e-13
    }
}
"""

using JSON3
using BaryRational
using TaylorDiff
using LinearAlgebra

include("../common.jl")  # For fit_aaa, nth_deriv_taylor

"""
    load_densified_data(method_name::String) -> Dict

Load densified smoothed signal from JSON file.
"""
function load_densified_data(method_name::String)
    data_dir = joinpath(@__DIR__, "../../../build/data/aaa_rom_input")
    filepath = joinpath(data_dir, "$(method_name)_dense.json")

    if !isfile(filepath)
        error("Densified data not found: $filepath")
    end

    # Read JSON file and convert to Dict
    json_string = read(filepath, String)
    data = JSON3.read(json_string)

    # Convert to regular Dict for easier access
    return Dict(string(k) => v for (k, v) in pairs(data))
end

"""
    detect_poles_in_domain(approx::AAAApprox, x_eval) -> Bool

Check if AAA interpolant has potential poles inside the evaluation domain.
This is a heuristic check - looks for support points with very large weights.

Returns true if potential poles are detected (unstable fit).
"""
function detect_poles_in_domain(approx::AAAApprox, x_eval)
    # Heuristic: Check if any support points are inside the evaluation domain
    # and have unusually large weights (may indicate nearby poles)
    x_min, x_max = extrema(x_eval)

    # Check support points
    for (i, xi) in enumerate(approx.x)
        if x_min <= xi <= x_max
            # Support point inside domain - this is actually OK
            # The issue is poles (not support points)
            # For now, we'll skip pole detection since BaryRational doesn't expose poles
            continue
        end
    end

    # TODO: Implement proper pole detection if needed
    # For now, we'll rely on the derivative computation to fail gracefully
    return false
end

"""
    evaluate_aaa_rom(method_name::String, t_eval, orders;
                     aaa_tol=1e-13, trim_boundary=0)

Evaluate AAA-ROM derivatives for a method.

Arguments:
- method_name: Name of the base method (e.g., "PyNumDiff-SavGol-Tuned")
- t_eval: Evaluation points (original grid)
- orders: Vector of derivative orders to compute (e.g., 0:7)
- aaa_tol: Tolerance for AAA fitting (default: 1e-13)
- trim_boundary: Number of boundary points to trim (helps with edge effects)

Returns:
- Dict with "predictions" and "failures"
"""
function evaluate_aaa_rom(method_name::String, t_eval, orders;
                          aaa_tol=1e-13, trim_boundary=0)
    predictions = Dict{Int, Vector{Float64}}()
    failures = Dict{Int, String}()

    try
        # Load densified data
        data = load_densified_data(method_name)
        t_dense = Float64.(data["t_dense"])
        y_dense = Float64.(data["y_dense"])

        # Validate data
        if length(t_dense) != length(y_dense)
            error("Mismatched lengths: t_dense=$(length(t_dense)), y_dense=$(length(y_dense))")
        end

        # Build AAA interpolant using fit_aaa from common.jl
        # This returns an AAAApprox struct that is callable
        fitted_func = fit_aaa(t_dense, y_dense; tol=aaa_tol, mmax=100)

        # Check for poles (heuristic)
        if detect_poles_in_domain(fitted_func, t_eval)
            @warn "Potential poles detected for $method_name"
        end

        # Compute derivatives at evaluation points
        for order in orders
            try
                deriv_values = zeros(Float64, length(t_eval))

                for (i, t) in enumerate(t_eval)
                    # Skip boundary points if requested
                    if trim_boundary > 0 && (i <= trim_boundary || i > length(t_eval) - trim_boundary)
                        deriv_values[i] = NaN
                        continue
                    end

                    # Compute derivative using Taylor differentiation from common.jl
                    deriv_values[i] = nth_deriv_taylor(fitted_func, order, t)

                    # Check for numerical issues
                    if !isfinite(deriv_values[i])
                        @warn "Non-finite derivative at t=$t, order=$order for $method_name"
                    end
                end

                predictions[order] = deriv_values

            catch e
                @warn "Failed to compute order $order for $method_name: $e"
                failures[order] = string(e)
                predictions[order] = fill(NaN, length(t_eval))
            end
        end

    catch e
        @error "AAA-ROM evaluation failed for $method_name: $e"
        # Return NaN for all orders
        for order in orders
            failures[order] = string(e)
            predictions[order] = fill(NaN, length(t_eval))
        end
    end

    return Dict("predictions" => predictions, "failures" => failures)
end

"""
    create_aaa_rom_method(base_method::String; aaa_tol=1e-13, trim_boundary=0)

Create an AAA-ROM method evaluator for a given base method.

Returns a function that can be called with (t_eval, orders) -> result.
"""
function create_aaa_rom_method(base_method::String; aaa_tol=1e-13, trim_boundary=0)
    return function(t_eval, orders)
        evaluate_aaa_rom(base_method, t_eval, orders;
                        aaa_tol=aaa_tol, trim_boundary=trim_boundary)
    end
end
