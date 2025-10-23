"""
Finite Difference Methods for Derivative Estimation

Implements Central-FD: Simple central finite difference estimator (baseline/comparison).
"""

# Load common utilities
include("../common.jl")

# ============================================================================
# Central Finite Differences
# ============================================================================

"""
	fit_finite_diff(x, y)

Simple finite difference estimator (for comparison/baseline).
"""
function fit_finite_diff(x, y)
	h = x[2] - x[1]  # Assume uniform spacing

	function fd_func(z, order)
		if order == 0
			# Nearest neighbor interpolation
			idx = argmin(abs.(x .- z))
			return y[idx]
		elseif order == 1
			# Central difference
			idx = argmin(abs.(x .- z))
			if idx == 1
				return (-3*y[1] + 4*y[2] - y[3]) / (2*h)
			elseif idx == length(x)
				return (y[end-2] - 4*y[end-1] + 3*y[end]) / (2*h)
			else
				return (y[idx+1] - y[idx-1]) / (2*h)
			end
		else
			return NaN  # Higher orders not implemented
		end
	end

	return fd_func
end


# ============================================================================
# Method Evaluator (Standard API)
# ============================================================================

"""
	evaluate_central_fd(x, y, x_eval, orders; params=Dict())

Evaluate Central-FD method: Simple central finite difference estimator.
NOTE: Only supports orders 0 and 1.
"""
function evaluate_central_fd(
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
		# Fit finite difference function
		fitted_func = fit_finite_diff(x, y)

		# Special handling for FD (has built-in derivative computation)
		for order in orders
			try
				predictions[order] = [fitted_func(xi, order) for xi in x_eval]
			catch e
				predictions[order] = fill(NaN, length(x_eval))
				failures[order] = string(e)
			end
		end

		timing = time() - t_start
		return MethodResult("Central-FD", "Finite Difference", predictions, failures, timing, true)

	catch e
		@warn "Central-FD failed" exception=e
		timing = time() - t_start
		return MethodResult("Central-FD", "Error", Dict(), Dict(0 => string(e)), timing, false)
	end
end
