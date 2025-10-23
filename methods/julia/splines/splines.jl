"""
Spline Methods for Derivative Estimation

Implements Dierckx smoothing spline with native derivative support.
"""

# Load common utilities
include("../common.jl")

using Dierckx

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
		# Fit Dierckx spline
		noise_level = get(params, :noise_level, 0.0)
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
