"""
ROM (Reduced Order Model) Wrapper using ApproxFun

This module provides a generic framework for computing higher-order derivatives
from ANY smoothing method using ApproxFun (Chebyshev polynomial approximation).

Workflow:
1. Any method produces densified smoothed signals (~1000 points)
2. Save to build/data/rom_input/<method>_dense.json
3. Julia reads data, builds ApproxFun function
4. Compute derivatives via ApproxFun's derivative operator

Data Format (JSON):
{
    "method_name": "PyNumDiff-SavGol-Tuned",
    "t_dense": [0.0, 0.001, 0.002, ...],  # Dense time grid (~1000 points)
    "y_dense": [1.2, 1.3, 1.4, ...],       # Smoothed signal on dense grid
    "t_eval": [0.0, 0.01, 0.02, ...],      # Original evaluation points
    "metadata": {
        "n_dense": 1000,
        "source_method": "...",
    }
}
"""

using JSON3
using ApproxFun

"""
    load_densified_data(method_name::String) -> Dict

Load densified smoothed signal from JSON file.
"""
function load_densified_data(method_name::String)
    data_dir = joinpath(@__DIR__, "../../../build/data/rom_input")
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
    fit_approxfun_from_data(t_dense::Vector, y_dense::Vector) -> Fun

Create ApproxFun function from data points.

ApproxFun will automatically:
- Choose optimal Chebyshev basis
- Determine number of coefficients
- Handle domain transformation
"""
function fit_approxfun_from_data(t_dense::Vector{T}, y_dense::Vector{T}) where {T<:Real}
    # Get domain
    t_min, t_max = extrema(t_dense)
    domain = t_min..t_max

    # Create Fun by fitting data
    # ApproxFun will use Chebyshev basis and determine coefficients automatically
    # This uses least squares fitting on Chebyshev points
    S = Chebyshev(domain)

    # Option 1: Use interpolation if data is on Chebyshev points (unlikely)
    # Option 2: Use least squares fit (more general)
    # We'll use a simple approach: create a function from the data via interpolation

    # For now, use a simple polynomial fit approach
    # ApproxFun will determine the degree automatically
    f = Fun(S, ApproxFun.transform(S, y_dense))

    return f
end

"""
    evaluate_rom(method_name::String, t_eval, orders; trim_boundary=0, max_degree=30)

Evaluate ROM derivatives for a method using ApproxFun.

Arguments:
- method_name: Name of the base method (e.g., "PyNumDiff-SavGol-Tuned")
- t_eval: Evaluation points (original grid)
- orders: Vector of derivative orders to compute (e.g., 0:7)
- trim_boundary: Number of boundary points to trim (helps with edge effects)
- max_degree: Maximum polynomial degree for least squares fit (default: 30)

Returns:
- Dict with "predictions" and "failures"
"""
function evaluate_rom(method_name::String, t_eval, orders; trim_boundary=0, max_degree=30)
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

        # Get domain and create ApproxFun space
        t_min, t_max = extrema(t_dense)
        S = Chebyshev(t_min..t_max)

        # Fit ApproxFun function using least squares with controlled degree
        # This prevents overfitting noise and keeps derivatives stable

        # Create basis functions
        basis = [Fun(S, [zeros(k-1); 1.0]) for k in 1:max_degree+1]

        # Least squares fit: Evaluate basis at data points
        A = zeros(length(t_dense), max_degree + 1)
        for (i, ti) in enumerate(t_dense)
            for (j, φ) in enumerate(basis)
                A[i, j] = φ(ti)
            end
        end

        # Solve least squares
        coeffs = A \ y_dense

        # Create Fun from coefficients
        fitted_func = Fun(S, coeffs)

        println("  ApproxFun fit: degree $max_degree ($(length(coefficients(fitted_func))) coefficients)")

        # Compute derivatives at evaluation points
        for order in orders
            try
                deriv_values = zeros(Float64, length(t_eval))

                # Get the derivative function
                if order == 0
                    f_deriv = fitted_func
                else
                    # ApproxFun has elegant derivative operator
                    f_deriv = fitted_func
                    for i in 1:order
                        f_deriv = f_deriv'  # Derivative operator
                    end
                end

                # Evaluate at all points
                for (i, t) in enumerate(t_eval)
                    # Skip boundary points if requested
                    if trim_boundary > 0 && (i <= trim_boundary || i > length(t_eval) - trim_boundary)
                        deriv_values[i] = NaN
                        continue
                    end

                    # Evaluate derivative
                    deriv_values[i] = f_deriv(t)

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
        @error "ROM evaluation failed for $method_name: $e"
        # Return NaN for all orders
        for order in orders
            failures[order] = string(e)
            predictions[order] = fill(NaN, length(t_eval))
        end
    end

    return Dict("predictions" => predictions, "failures" => failures)
end

"""
    create_rom_method(base_method::String; trim_boundary=0, max_degree=30)

Create a ROM method evaluator for a given base method.

Returns a function that can be called with (t_eval, orders) -> result.
"""
function create_rom_method(base_method::String; trim_boundary=0, max_degree=30)
    return function(t_eval, orders)
        evaluate_rom(base_method, t_eval, orders; trim_boundary=trim_boundary, max_degree=max_degree)
    end
end
