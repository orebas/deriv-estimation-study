"""
Filtering Methods for Derivative Estimation

Implements Savitzky-Golay filter for smoothing and differentiation.
"""

# Load common utilities
include("../common.jl")

# ============================================================================
# Savitzky-Golay Filter
# ============================================================================

"""Simple SG smoothing (Vandermonde matrix approach)"""
function savitzky_golay_smooth(y, window, polyorder)
	n = length(y)
	half_window = window ÷ 2

	# Build Vandermonde matrix for polynomial fitting
	pos = collect((-half_window):half_window)
	A = zeros(window, polyorder + 1)
	for i in 1:window
		for j in 0:polyorder
			A[i, j+1] = pos[i]^j
		end
	end

	# Compute filter coefficients (SG filter weights)
	# The filter coefficients are: (A^T A)^{-1} A^T e_center
	# where e_center is the unit vector selecting the center point
	e_center = zeros(window)
	e_center[half_window+1] = 1.0

	# Solve normal equations: (A^T A) c = A^T e_center
	# Then filter weights are: w = A * c
	c = (A' * A) \ (A' * e_center)
	filter_weights = A * c

	# Apply filter
	smoothed = similar(y)
	for i in 1:n
		if i <= half_window || i > n - half_window
			smoothed[i] = y[i]  # Keep boundary points
		else
			window_data = y[(i-half_window):(i+half_window)]
			smoothed[i] = dot(filter_weights, window_data)
		end
	end

	return smoothed
end

"""
	fit_savitzky_golay(x, y; window=11, polyorder=5)

Savitzky-Golay filter for smoothing and differentiation.
"""
function fit_savitzky_golay(x, y; window = 11, polyorder = 5)
	# Ensure window is odd
	window = isodd(window) ? window : window + 1
	window = min(window, length(y))

	# Smooth the data
	smoothed = savitzky_golay_smooth(y, window, polyorder)

	# Return interpolator
	return z -> begin
		idx = argmin(abs.(x .- z))
		return smoothed[idx]
	end
end


# ============================================================================
# Method Evaluator (Standard API)
# ============================================================================

"""
	evaluate_savitzky_golay(x, y, x_eval, orders; params=Dict())

Evaluate Savitzky-Golay method: Local polynomial smoothing with AD-based derivatives.
"""
function evaluate_savitzky_golay(
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
		# Get parameters
		window = get(params, :sg_window, min(21, length(x)))
		polyorder = get(params, :sg_polyorder, 5)

		# Fit Savitzky-Golay filter
		fitted_func = fit_savitzky_golay(x, y; window = window, polyorder = polyorder)

		# Use TaylorDiff for automatic differentiation (all orders)
		for order in orders
			try
				predictions[order] = [nth_deriv_taylor(fitted_func, order, xi) for xi in x_eval]
			catch e
				predictions[order] = fill(NaN, length(x_eval))
				failures[order] = string(e)
			end
		end

		timing = time() - t_start
		return MethodResult("Savitzky-Golay", "Local Polynomial", predictions, failures, timing, true)

	catch e
		@warn "Savitzky-Golay failed" exception=e
		timing = time() - t_start
		return MethodResult("Savitzky-Golay", "Error", Dict(), Dict(0 => string(e)), timing, false)
	end
end
