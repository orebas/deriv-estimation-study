"""
ApproxFun Extension for Limited-Order Methods

Extends methods that only provide low-order derivatives (0-3) to higher orders (up to 7)
using ApproxFun.jl's spectral differentiation capabilities.

Based on guidance from multiple model consultations:
- Uses adaptive fitting with tolerance control (not fixed degree)
- Monitors coefficient decay to validate smoothness
- Careful handling of boundary effects
- Acts as secondary precision smoothing after initial denoising
"""

using ApproxFun
using Statistics
using LinearAlgebra
using Interpolations

"""
    extend_with_approxfun(t, y_smooth, t_eval, max_order=7;
                         tol=1e-8, max_coeffs=500,
                         check_decay=true, trim_boundary=0.05)

Extend a smoothed signal to compute higher-order derivatives using ApproxFun.

# Arguments
- `t`: Time points of smoothed signal
- `y_smooth`: Smoothed signal values (already denoised by PyNumDiff etc.)
- `t_eval`: Points at which to evaluate derivatives
- `max_order`: Maximum derivative order to compute (default 7)

# Keywords
- `tol`: Tolerance for adaptive fitting (default 1e-8)
- `max_coeffs`: Maximum number of Chebyshev coefficients (default 500)
- `check_decay`: Whether to validate coefficient decay (default true)
- `trim_boundary`: Fraction of boundary to mark as unreliable (default 0.05)

# Returns
Dictionary with:
- `predictions`: Dict of order => derivative values
- `metadata`: Information about fit quality
- `warnings`: Any issues detected
"""
function extend_with_approxfun(t::Vector{Float64},
                              y_smooth::Vector{Float64},
                              t_eval::Vector{Float64},
                              max_order::Int=7;
                              tol::Float64=1e-8,
                              max_coeffs::Int=500,
                              check_decay::Bool=true,
                              trim_boundary::Float64=0.05)

    warnings = String[]

    # Step 1: Create Chebyshev space on the data domain
    t_min, t_max = extrema(t)
    domain = t_min..t_max
    S = Chebyshev(domain)

    # Step 2: Fit with adaptive tolerance
    # Using ApproxFun's built-in discrete fitting as recommended

    # Create an interpolation function first
    itp = linear_interpolation(t, y_smooth, extrapolation_bc=Flat())

    # Now create Fun from the interpolation function
    # ApproxFun will adaptively choose coefficients
    # We control accuracy by specifying max coefficients
    # Default constructor aims for machine precision

    # First, get an initial fit
    f_initial = Fun(x -> itp(x), domain)
    coeffs_initial = coefficients(f_initial)
    n_coeffs_initial = length(coeffs_initial)

    # Find how many coefficients we need for desired tolerance
    # Coefficients should decay exponentially for smooth functions
    n_keep = findfirst(i -> abs(coeffs_initial[i]) < tol, 1:n_coeffs_initial)
    if isnothing(n_keep)
        n_keep = min(n_coeffs_initial, max_coeffs)
    else
        n_keep = min(n_keep, max_coeffs)
    end

    # Truncate to desired tolerance
    coeffs = coeffs_initial[1:n_keep]
    f = Fun(S, coeffs)
    n_coeffs = n_keep

    if n_coeffs == max_coeffs
        push!(warnings, "Hit coefficient limit of $max_coeffs")
    end

    # Step 3: Validate coefficient decay if requested
    coeffs = coefficients(f)
    n_coeffs = length(coeffs)

    decay_metrics = Dict{String, Any}()
    if check_decay && n_coeffs > 10
        # Check exponential decay of coefficients
        log_coeffs = log10.(abs.(coeffs) .+ 1e-16)

        # Fit linear trend to log coefficients (should be negative slope)
        X = [ones(n_coeffs) collect(1:n_coeffs)]
        β = X \ log_coeffs
        decay_rate = -β[2]  # Negative of slope (positive = good decay)

        decay_metrics["n_coefficients"] = n_coeffs
        decay_metrics["decay_rate"] = decay_rate
        decay_metrics["final_coeff_magnitude"] = abs(coeffs[end])

        # Warning if poor decay
        if decay_rate < 0.05  # Less than 0.05 decade per coefficient
            push!(warnings, "Poor coefficient decay (rate=$decay_rate), high derivatives may be noisy")
        end
    end

    # Step 4: Compute derivatives
    predictions = Dict{Int, Vector{Float64}}()

    # Order 0: function values
    predictions[0] = [f(t) for t in t_eval]

    # Higher orders: successive differentiation
    f_deriv = f
    for order in 1:max_order
        f_deriv = f_deriv'  # Spectral differentiation
        predictions[order] = [f_deriv(t) for t in t_eval]
    end

    # Step 5: Mark boundary regions as unreliable
    if trim_boundary > 0
        n_eval = length(t_eval)
        n_trim = ceil(Int, trim_boundary * n_eval)

        boundary_indices = vcat(1:n_trim, (n_eval-n_trim+1):n_eval)

        # Add warning about boundaries for high derivatives
        if max_order >= 5
            push!(warnings, "Derivatives near boundaries (first/last $(n_trim) points) may be unreliable for orders ≥ 5")
        end
    end

    # Step 6: Compute fit quality metrics
    # Residual at original points
    y_fitted = [f(t_i) for t_i in t]
    residual = y_smooth .- y_fitted
    rmse_fit = sqrt(mean(residual.^2))
    max_residual = maximum(abs.(residual))

    metadata = Dict{String, Any}(
        "n_coefficients" => n_coeffs,
        "tolerance_used" => tol,
        "rmse_fit" => rmse_fit,
        "max_residual" => max_residual,
        "decay_metrics" => decay_metrics,
        "domain" => (t_min, t_max)
    )

    return Dict(
        "predictions" => predictions,
        "metadata" => metadata,
        "warnings" => warnings
    )
end


"""
    validate_smoothness_for_order(y_smooth, target_order; dt=1.0)

Estimate whether a smoothed signal is smooth enough for a given derivative order.
Uses finite difference approximations to check if derivatives are well-behaved.

Returns (is_smooth, estimated_noise_level)
"""
function validate_smoothness_for_order(y_smooth::Vector{Float64},
                                      target_order::Int;
                                      dt::Float64=1.0)
    n = length(y_smooth)

    # Use progressively higher-order finite differences as proxy
    current = y_smooth

    for order in 1:min(target_order, 4)  # Don't go beyond 4th order FD
        if length(current) < 5
            return (false, Inf)  # Not enough points
        end

        # Central differences (rough approximation)
        next = zeros(length(current) - 2)
        for i in 2:(length(current)-1)
            next[i-1] = (current[i+1] - current[i-1]) / (2*dt)
        end
        current = next
    end

    # Check variability in estimated derivative
    if length(current) > 0
        noise_estimate = std(diff(current))
        signal_estimate = std(current)

        snr = signal_estimate / (noise_estimate + 1e-10)

        # Rule of thumb: need SNR > 10^(order/2) for decent derivatives
        required_snr = 10^(target_order/2)

        return (snr > required_snr, noise_estimate)
    else
        return (false, Inf)
    end
end


"""
    wrap_pynumdiff_with_approxfun(method_name, t, y_noisy, t_eval,
                                 pynumdiff_result; kwargs...)

Wrapper specifically for PyNumDiff methods that extends their limited order output.

Expects pynumdiff_result to have structure:
- predictions[0]: smoothed signal
- predictions[1]: first derivative (optional)
"""
function wrap_pynumdiff_with_approxfun(method_name::String,
                                      t::Vector{Float64},
                                      y_noisy::Vector{Float64},
                                      t_eval::Vector{Float64},
                                      pynumdiff_result::Dict;
                                      max_order::Int=7,
                                      adaptive_tol::Bool=true,
                                      kwargs...)

    # Extract smoothed signal from PyNumDiff
    if !haskey(pynumdiff_result, "predictions") || !haskey(pynumdiff_result["predictions"], 0)
        return Dict(
            "predictions" => Dict{Int, Vector{Float64}}(),
            "failures" => Dict(0 => "No smoothed signal from PyNumDiff"),
            "metadata" => Dict()
        )
    end

    y_smooth = pynumdiff_result["predictions"][0]

    # Adaptive tolerance based on method characteristics
    if adaptive_tol
        if occursin("Butter", method_name) || occursin("Gaussian", method_name)
            # These methods produce very smooth output
            tol = 1e-9
        elseif occursin("Kalman", method_name)
            # Kalman is extremely smooth
            tol = 1e-10
        elseif occursin("TV", method_name)
            # TV methods may have slight discontinuities in derivatives
            tol = 1e-7
        else
            tol = 1e-8  # Default
        end
    else
        tol = get(kwargs, :tol, 1e-8)
    end

    # Check if signal is smooth enough for high derivatives
    is_smooth, noise_level = validate_smoothness_for_order(y_smooth, max_order)

    if !is_smooth && max_order > 5
        # Consider reducing max_order or warning
        @warn "Signal may not be smooth enough for order $max_order derivatives"
    end

    # Extend using ApproxFun (filter out unsupported kwargs)
    supported_kwargs = [:max_coeffs, :check_decay, :trim_boundary]
    filtered_kwargs = filter(kv -> kv.first in supported_kwargs, kwargs)

    result = extend_with_approxfun(t, y_smooth, t_eval, max_order;
                                  tol=tol, filtered_kwargs...)

    # Merge with original PyNumDiff results for orders 0-1 if available
    # (they might be more accurate for low orders)
    if get(kwargs, :prefer_original_low_orders, true)
        for order in 0:1
            if haskey(pynumdiff_result["predictions"], order)
                # Keep original for these orders
                result["predictions"][order] = pynumdiff_result["predictions"][order]
            end
        end
    end

    # Add method info to metadata
    result["metadata"]["base_method"] = method_name
    result["metadata"]["extension_method"] = "ApproxFun"
    result["metadata"]["noise_estimate"] = noise_level

    return result
end


# Export main functions
export extend_with_approxfun, validate_smoothness_for_order, wrap_pynumdiff_with_approxfun