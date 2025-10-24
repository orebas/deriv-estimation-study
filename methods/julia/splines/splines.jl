"""
Spline Methods for Derivative Estimation

Implements Dierckx smoothing spline with native derivative support.
"""

# Load common utilities
include("../common.jl")

using Dierckx
using GeneralizedSmoothingSplines
using ForwardDiff

# Import noise estimation module
include(joinpath(@__DIR__, "..", "..", "..", "src", "hyperparameter_selection.jl"))
using .HyperparameterSelection

# ============================================================================
# Dierckx Spline Fitting
# ============================================================================

"""
	fit_dierckx_spline(x, y; k=5, s=nothing, noise_level=0.0)

Fit smoothing spline using Dierckx.
"""
function fit_dierckx_spline(x, y; k = 5, s = nothing, noise_level = 0.0)
	# Auto-select smoothing parameter if not provided
	if isnothing(s)
		n = length(x)
		σ = noise_level * std(y)
		s = n * σ^2  # GCV-like heuristic
	end

	# Fit spline
	spl = Spline1D(x, y; k = k, s = s)

	# Return callable
	return z -> evaluate(spl, z), spl  # Return both func and spline object
end


# ============================================================================
# Method Evaluator (Standard API)
# ============================================================================

"""
	evaluate_dierckx(x, y, x_eval, orders; params=Dict())

Evaluate Dierckx-5 method: Quintic spline with native derivative support.
"""
function evaluate_dierckx(
	x::Vector{Float64},
	y::Vector{Float64},
	x_eval::Vector{Float64},
	orders::Vector{Int};
	params = Dict()
)
	t_start = time()
	predictions = Dict{Int, Vector{Float64}}()
	failures = Dict{Int, String}()

	try
		# Estimate noise from data using wavelet MAD (fair comparison with other methods)
		# Wavelet is robust to signal curvature, unlike diff2 which overestimates for smooth signals
		noise_level = HyperparameterSelection.estimate_noise_wavelet(y)
		_, spl = fit_dierckx_spline(x, y; k = 5, noise_level = noise_level)

		# Dierckx has native derivative support
		for order in orders
			try
				if order == 0
					predictions[order] = [evaluate(spl, xi) for xi in x_eval]
				else
					predictions[order] = [Dierckx.derivative(spl, xi, nu = order) for xi in x_eval]
				end
			catch e
				predictions[order] = fill(NaN, length(x_eval))
				failures[order] = string(e)
			end
		end

		timing = time() - t_start
		return MethodResult("Dierckx-5", "Spline", predictions, failures, timing, true)

	catch e
		@warn "Dierckx-5 failed" exception=e
		timing = time() - t_start
		return MethodResult("Dierckx-5", "Error", Dict(), Dict(0 => string(e)), timing, false)
	end
end


# ============================================================================
# GeneralizedSmoothingSplines (GSS) - High-order smoothing splines
# ============================================================================

"""
	fit_gss_spline(x, y; max_order=7, noise_level=0.0)

Fit smoothing spline using GeneralizedSmoothingSplines.
Uses degree 2p-1 where p is chosen to support derivatives up to max_order.
"""
function fit_gss_spline(x, y; max_order = 7, noise_level = 0.0)
	# Choose p to support max_order derivatives
	# For order n, need degree >= n+1, so 2p-1 >= n+1, thus p >= (n+2)/2
	p = ceil(Int, (max_order + 2) / 2)

	# Create model
	model = GeneralizedSmoothingSplines.SmoothingSpline(p = p, lambda = 0.01)

	# Tune lambda using marginal likelihood if noise level is available
	if noise_level > 0.0
		try
			GeneralizedSmoothingSplines.tune!(model, x, y; show_trace = false)
		catch e
			# If tuning fails, use default lambda
			@warn "GSS tuning failed, using default lambda" exception = e
		end
	end

	# Fit the model
	fitresult, _, _ = GeneralizedSmoothingSplines.MMI.fit(model, 0, x, y)

	# Return prediction function
	pred_fn = function(t::T) where T
		result_vec = GeneralizedSmoothingSplines.MMI.predict(model, fitresult, T[t])
		return result_vec[1]
	end

	return pred_fn, model, fitresult
end

"""
	nth_derivative_forwarddiff(f, x, n)

Compute n-th derivative using recursive ForwardDiff.
Required because TaylorDiff doesn't work with GSS's complex predict implementation.
"""
function nth_derivative_forwarddiff(f, x, n)
	if n == 0
		return f(x)
	elseif n == 1
		return ForwardDiff.derivative(f, x)
	else
		return ForwardDiff.derivative(t -> nth_derivative_forwarddiff(f, t, n - 1), x)
	end
end

"""
	evaluate_gss(x, y, x_eval, orders; params=Dict())

Evaluate GSS method: High-order smoothing spline with automatic lambda tuning.
Supports arbitrary derivative orders using recursive ForwardDiff.
"""
function evaluate_gss(
	x::Vector{Float64},
	y::Vector{Float64},
	x_eval::Vector{Float64},
	orders::Vector{Int};
	params = Dict()
)
	t_start = time()
	predictions = Dict{Int, Vector{Float64}}()
	failures = Dict{Int, String}()

	try
		# Estimate noise from data using wavelet MAD
		noise_level = HyperparameterSelection.estimate_noise_wavelet(y)

		# Fit spline with appropriate degree for max order
		max_order = maximum(orders)
		pred_fn, model, fitresult = fit_gss_spline(x, y; max_order = max_order, noise_level = noise_level)

		# Compute derivatives using recursive ForwardDiff
		for order in orders
			try
				predictions[order] = [nth_derivative_forwarddiff(pred_fn, xi, order) for xi in x_eval]
			catch e
				predictions[order] = fill(NaN, length(x_eval))
				failures[order] = string(e)
			end
		end

		timing = time() - t_start
		return MethodResult("GSS", "Spline", predictions, failures, timing, true)

	catch e
		@warn "GSS failed" exception = e
		timing = time() - t_start
		return MethodResult("GSS", "Error", Dict(), Dict(0 => string(e)), timing, false)
	end
end
