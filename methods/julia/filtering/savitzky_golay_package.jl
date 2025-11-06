"""
Savitzky-Golay derivative estimation using SavitzkyGolay.jl package.

Implements three methods based on expert consultation (GPT-5 + Gemini-2.5-Pro):
1. Fixed: Robust baseline with fixed physical window
2. Hybrid: Gentle adaptive adjustment around fixed baseline
3. Adaptive: Pure theoretical approach for comparison

Key insight: Optimal window size is a physical quantity h (in domain units),
not a sample count w. This makes methods generalizable across discretizations.
"""

using SavitzkyGolay
using Statistics
using LinearAlgebra

# Import noise estimation from HyperparameterSelection
import ..HyperparameterSelection

# Import MethodResult from parent common module
# (already included in julia_methods_integrated.jl)
using ..Main: MethodResult

"""
    get_physical_window(polyorder::Int, deriv_order::Int) -> Float64

Get calibrated physical window size h from GPT-5 recommendation table.
These values are tuned for smooth ODE trajectories at N≈251.
"""
function get_physical_window(polyorder::Int, deriv_order::Int)
    # Calibration table from GPT-5
    h_table = Dict(
        # p = 7
        (7, 0) => 0.16, (7, 1) => 0.16, (7, 2) => 0.16,
        (7, 3) => 0.18, (7, 4) => 0.18,
        (7, 5) => 0.20,
        (7, 6) => 0.22, (7, 7) => 0.22,
        # p = 9
        (9, 0) => 0.15, (9, 1) => 0.15, (9, 2) => 0.15,
        (9, 3) => 0.17, (9, 4) => 0.17,
        (9, 5) => 0.19,
        (9, 6) => 0.21, (9, 7) => 0.21,
        # p = 11 (for very low noise only)
        (11, 0) => 0.14, (11, 1) => 0.14, (11, 2) => 0.14,
        (11, 3) => 0.16, (11, 4) => 0.16,
        (11, 5) => 0.18,
        (11, 6) => 0.20, (11, 7) => 0.20,
    )

    # Lookup with fallback
    if haskey(h_table, (polyorder, deriv_order))
        return h_table[(polyorder, deriv_order)]
    else
        # Fallback: increase h for higher derivatives
        return 0.16 + 0.01 * deriv_order
    end
end

"""
    select_polyorder(deriv_order::Int, noise_relative::Float64 = 0.0) -> Int

Select polynomial order based on derivative order and relative noise level.
Default: p = clamp(r + 2, 5, 9)
"""
function select_polyorder(deriv_order::Int, noise_relative::Float64 = 0.0)
    # Base tier: p = clamp(r + 2, 5, 9)
    p = clamp(deriv_order + 2, 5, 9)

    # Special case: very low noise + high derivative → allow p=11
    if noise_relative > 0.0 && noise_relative ≤ 5e-6 && deriv_order ≥ 5
        p = min(deriv_order + 4, 11)
    end

    # Special case: high noise → reduce to minimum
    if noise_relative > 0.0 && noise_relative ≥ 1e-2 && deriv_order ≥ 3
        p = max(deriv_order + 1, 5)
    end

    return p
end

"""
    compute_window_size(h_physical::Float64, dx::Float64, n::Int, polyorder::Int) -> Int

Convert physical window h to discrete window size w with constraints.
"""
function compute_window_size(h_physical::Float64, dx::Float64, n::Int, polyorder::Int)
    # Convert physical window to discrete samples
    w_ideal = 2 * floor(Int, h_physical / dx) + 1

    # Apply constraints
    w_min = polyorder + 3  # Numerical stability (not just p+2)
    w_max = n ÷ 3          # Cap at 1/3 of data size

    w = clamp(w_ideal, w_min, min(w_max, n))

    # Ensure odd
    w = isodd(w) ? w : w - 1

    return w
end

"""
    compute_boundary_discard(window_size::Int, deriv_order::Int) -> Int

Compute number of boundary points to discard (GPT-5 derivative-dependent approach).
Higher derivatives need more aggressive boundary removal.
"""
function compute_boundary_discard(window_size::Int, deriv_order::Int)
    hw = (window_size - 1) ÷ 2

    if deriv_order ≤ 2
        return ceil(Int, 0.5 * hw)  # Discard ~50% of half-window
    elseif deriv_order ≤ 4
        return ceil(Int, 0.75 * hw) # Discard ~75% of half-window
    else  # deriv_order ≥ 5
        return hw                    # Discard full half-window
    end
end

"""
    estimate_roughness_4th_order(y::Vector{Float64}, dx::Float64) -> Float64

Estimate signal roughness using 4th-order finite differences (Rice 1984).
"""
function estimate_roughness_4th_order(y::Vector{Float64}, dx::Float64)
    d4 = y
    for _ in 1:4
        d4 = diff(d4)
    end
    ρ_hat = sqrt(mean(d4.^2)) / (dx^4 + 1e-24)
    return ρ_hat
end

"""
    linear_interpolate_derivatives(x_valid::Vector{Float64}, y_valid::Vector{Float64},
                                   x_eval::Vector{Float64}) -> Vector{Float64}

Linear interpolation to evaluation points with flat extrapolation at boundaries.
"""
function linear_interpolate_derivatives(x_valid::Vector{Float64}, y_valid::Vector{Float64},
                                        x_eval::Vector{Float64})
    result = zeros(length(x_eval))

    for (i, xi) in enumerate(x_eval)
        if xi < x_valid[1]
            # Flat extrapolation (use first valid value)
            result[i] = y_valid[1]
        elseif xi > x_valid[end]
            # Flat extrapolation (use last valid value)
            result[i] = y_valid[end]
        else
            # Linear interpolation
            idx = searchsortedfirst(x_valid, xi)
            if idx == 1
                result[i] = y_valid[1]
            else
                t = (xi - x_valid[idx-1]) / (x_valid[idx] - x_valid[idx-1])
                result[i] = (1 - t) * y_valid[idx-1] + t * y_valid[idx]
            end
        end
    end

    return result
end

"""
    evaluate_savitzky_golay_package_fixed(x, y, x_eval, orders; params=Dict())

Method 1: SG-Package-Fixed
Fixed physical window approach (robust baseline).

Uses calibrated physical windows h(p,r) from GPT-5 recommendations,
converting to discrete windows w = round(h/dx) with proper constraints.

# Parameters
- No hyperparameters (all automatic based on calibration table)
"""
function evaluate_savitzky_golay_package_fixed(x::Vector{Float64}, y::Vector{Float64},
                                                x_eval::Vector{Float64}, orders::Vector{Int};
                                                params::Dict = Dict())
    t_start = time()
    n = length(x)
    dx = mean(diff(x))
    signal_scale = maximum(abs.(y))

    predictions = Dict{Int, Vector{Float64}}()
    failures = Dict{Int, String}()

    for deriv_order in orders
        try
            # Select polynomial order (simple tiered approach)
            p = clamp(deriv_order + 2, 5, 9)

            # Get physical window from calibration table
            h = get_physical_window(p, deriv_order)

            # Compute discrete window size
            w = compute_window_size(h, dx, n, p)

            # Call SavitzkyGolay.jl package
            # CRITICAL: rate = 1/dx (sampling rate, NOT spacing)
            result = savitzky_golay(y, w, p; deriv=deriv_order, rate=1/dx)

            # Discard boundaries (derivative-order dependent)
            m = compute_boundary_discard(w, deriv_order)
            valid_indices = (m+1):(n-m)

            # Interpolate to x_eval (only from valid interior points)
            deriv_eval = linear_interpolate_derivatives(
                x[valid_indices],
                result.y[valid_indices],
                x_eval
            )

            predictions[deriv_order] = deriv_eval

        catch e
            @warn "SG-Package-Fixed failed for order $deriv_order: $e"
            predictions[deriv_order] = fill(NaN, length(x_eval))
            failures[deriv_order] = string(e)
        end
    end

    timing = time() - t_start
    return MethodResult("SG-Package-Fixed", "Local Polynomial", predictions, failures, timing, true)
end

"""
    evaluate_savitzky_golay_package_hybrid(x, y, x_eval, orders; params=Dict())

Method 2: SG-Package-Hybrid
Gentle adaptive adjustment around fixed baseline (GPT-5 recommendation).

Blends fixed physical window with adaptive suggestion:
    h = 0.7*h_base + 0.3*h_adaptive
This provides stability while allowing response to extreme noise cases.

# Parameters
- No hyperparameters (all automatic)
"""
function evaluate_savitzky_golay_package_hybrid(x::Vector{Float64}, y::Vector{Float64},
                                                 x_eval::Vector{Float64}, orders::Vector{Int};
                                                 params::Dict = Dict())
    t_start = time()
    n = length(x)
    dx = mean(diff(x))
    signal_scale = maximum(abs.(y))

    predictions = Dict{Int, Vector{Float64}}()
    failures = Dict{Int, String}()

    # Estimate noise and roughness once (same for all orders)
    σ_hat = HyperparameterSelection.estimate_noise_wavelet(y)
    ρ_hat = estimate_roughness_4th_order(y, dx)

    # Blending parameter
    α = 0.3  # 30% adaptive, 70% fixed

    for deriv_order in orders
        try
            # Select polynomial order
            p = clamp(deriv_order + 2, 5, 9)

            # Get baseline physical window
            h_base = get_physical_window(p, deriv_order)

            # Compute adaptive suggestion using MISE formula
            c = h_base  # Use baseline as calibration constant
            σ_safe = max(σ_hat, 1e-24)
            ρ_safe = max(ρ_hat, 1e-24)
            exponent = 1.0 / (2*p + 3)
            h_adapt = c * (σ_safe^2 / ρ_safe^2)^exponent

            # Blend and clamp
            h = (1 - α) * h_base + α * h_adapt
            h = clamp(h, 0.10, 0.28)  # Physical window bounds

            # Convert to discrete window
            w = compute_window_size(h, dx, n, p)

            # Call package
            result = savitzky_golay(y, w, p; deriv=deriv_order, rate=1/dx)

            # Discard boundaries
            m = compute_boundary_discard(w, deriv_order)
            valid_indices = (m+1):(n-m)

            # Interpolate to x_eval
            deriv_eval = linear_interpolate_derivatives(
                x[valid_indices],
                result.y[valid_indices],
                x_eval
            )

            predictions[deriv_order] = deriv_eval

        catch e
            @warn "SG-Package-Hybrid failed for order $deriv_order: $e"
            predictions[deriv_order] = fill(NaN, length(x_eval))
            failures[deriv_order] = string(e)
        end
    end

    timing = time() - t_start
    return MethodResult("SG-Package-Hybrid", "Local Polynomial", predictions, failures, timing, true)
end

"""
    evaluate_savitzky_golay_package_adaptive(x, y, x_eval, orders; params=Dict())

Method 3: SG-Package-Adaptive
Pure theoretical MISE-based approach for comparison.

Uses adaptive formula directly with minimal fixed constraints.
Expected to work well at N=251 but may be unstable at extremes.

# Parameters
- No hyperparameters (all automatic)
"""
function evaluate_savitzky_golay_package_adaptive(x::Vector{Float64}, y::Vector{Float64},
                                                   x_eval::Vector{Float64}, orders::Vector{Int};
                                                   params::Dict = Dict())
    t_start = time()
    n = length(x)
    dx = mean(diff(x))

    predictions = Dict{Int, Vector{Float64}}()
    failures = Dict{Int, String}()

    # Estimate noise and roughness
    σ_hat = HyperparameterSelection.estimate_noise_wavelet(y)
    ρ_hat = estimate_roughness_4th_order(y, dx)

    for deriv_order in orders
        try
            # Select polynomial order
            p = clamp(deriv_order + 2, 5, 9)

            # Compute adaptive window (pure formula)
            c = get_physical_window(p, deriv_order)  # Use as scaling constant
            σ_safe = max(σ_hat, 1e-24)
            ρ_safe = max(ρ_hat, 1e-24)
            exponent = 1.0 / (2*p + 3)
            h_adapt = c * (σ_safe^2 / ρ_safe^2)^exponent

            # Apply hard bounds (prevent pathological cases)
            h = clamp(h_adapt, 0.05, 0.35)

            # Convert to discrete window
            w = compute_window_size(h, dx, n, p)

            # Call package
            result = savitzky_golay(y, w, p; deriv=deriv_order, rate=1/dx)

            # Discard boundaries
            m = compute_boundary_discard(w, deriv_order)
            valid_indices = (m+1):(n-m)

            # Interpolate to x_eval
            deriv_eval = linear_interpolate_derivatives(
                x[valid_indices],
                result.y[valid_indices],
                x_eval
            )

            predictions[deriv_order] = deriv_eval

        catch e
            @warn "SG-Package-Adaptive failed for order $deriv_order: $e"
            predictions[deriv_order] = fill(NaN, length(x_eval))
            failures[deriv_order] = string(e)
        end
    end

    timing = time() - t_start
    return MethodResult("SG-Package-Adaptive", "Local Polynomial", predictions, failures, timing, true)
end
