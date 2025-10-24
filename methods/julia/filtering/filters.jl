"""
Filtering Methods for Derivative Estimation

Implements Savitzky-Golay filter for smoothing and differentiation.
"""

# Load common utilities
include("../common.jl")

using LinearAlgebra

# ============================================================================
# Savitzky-Golay Filter - Derivative Estimation
# ============================================================================

"""
	savitzky_golay_coeffs(window::Int, polyorder::Int, deriv_order::Int=0)

Compute Savitzky-Golay filter coefficients for a given window size, polynomial order,
and derivative order.

This is the standard implementation used by scipy.signal.savgol_filter and other libraries.

# Arguments
- `window`: Window size (must be odd)
- `polyorder`: Order of polynomial fit
- `deriv_order`: Derivative order to compute (0 = smoothing only)

# Returns
Filter coefficients to apply via convolution

# References
- Savitzky, A., & Golay, M.J.E. (1964). Smoothing and differentiation of data by
  simplified least squares procedures. Analytical Chemistry, 36(8), 1627-1639.
"""
function savitzky_golay_coeffs(window::Int, polyorder::Int, deriv_order::Int = 0)
	# Validation
	isodd(window) || throw(ArgumentError("Window size must be odd, got $window"))
	polyorder >= deriv_order ||
		throw(ArgumentError("Polynomial order ($polyorder) must be >= derivative order ($deriv_order)"))
	window > polyorder || throw(ArgumentError("Window size ($window) must be > polyorder ($polyorder)"))

	half_window = window ÷ 2

	# Build Vandermonde matrix for polynomial fitting over integer positions
	pos = collect((-half_window):half_window)
	A = zeros(window, polyorder + 1)
	for i = 1:window, j = 0:polyorder
		A[i, j+1] = pos[i]^j
	end

	# The filter extracts the deriv_order-th derivative coefficient
	# For a polynomial Σ c_j x^j, the k-th derivative at x=0 is k! * c_k
	# We want coefficients that extract c_k from the least-squares fit

	# Solve (A'A) c = e_{k+1} where e_{k+1} selects the k-th coefficient
	e_k = zeros(polyorder + 1)
	e_k[deriv_order+1] = 1.0

	# Solve normal equations
	c = (A' * A) \ e_k

	# Filter weights are A * c, scaled by factorial(k) for derivative
	filter_weights = A * c * factorial(deriv_order)

	return filter_weights
end


"""
	apply_savitzky_golay_filter(y::Vector{Float64}, window::Int, polyorder::Int, deriv_order::Int, dx::Float64)

Apply Savitzky-Golay filter to compute derivatives at interior points.

# Arguments
- `y`: Data values
- `window`: Window size (must be odd)
- `polyorder`: Polynomial order
- `deriv_order`: Derivative order to compute
- `dx`: Grid spacing (for scaling derivatives)

# Returns
Vector of derivative values (NaN at boundaries where filter can't be applied)
"""
function apply_savitzky_golay_filter(
	y::Vector{Float64},
	window::Int,
	polyorder::Int,
	deriv_order::Int,
	dx::Float64
)
	n = length(y)
	half_window = window ÷ 2

	# Get filter coefficients
	weights = savitzky_golay_coeffs(window, polyorder, deriv_order)

	# Apply filter at each point
	result = fill(NaN, n)

	for i = 1:n
		if i <= half_window || i > n - half_window
			# Boundary points - cannot apply full window
			# Leave as NaN (will be handled by interpolation)
			continue
		end

		# Extract window of data
		window_data = y[(i-half_window):(i+half_window)]

		# Apply filter and scale by dx^deriv_order
		# (SG coefficients assume unit spacing, so we need to scale)
		result[i] = dot(weights, window_data) / (dx^deriv_order)
	end

	return result
end


# ============================================================================
# Method Evaluator (Standard API)
# ============================================================================

"""
	evaluate_savitzky_golay(x, y, x_eval, orders; params=Dict())

Evaluate Savitzky-Golay method using proper SG derivative filters.

This is the CORRECT implementation that matches scipy.signal.savgol_filter behavior.
Uses analytical SG derivative filter coefficients rather than AD on interpolators.

# Arguments
- `x`: Training x points (assumed approximately uniformly spaced)
- `y`: Training y values
- `x_eval`: Evaluation points
- `orders`: Derivative orders to compute
- `params`: Optional parameters
  - `:sg_window`: Window size (default: min(21, length(x)), must be odd)
  - `:sg_polyorder`: Polynomial order (default: 5)

# Returns
MethodResult with predictions for each derivative order
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
		window = isodd(window) ? window : window + 1  # Ensure odd
		window = min(window, length(x))
		polyorder = get(params, :sg_polyorder, 5)

		# Check for uniform spacing
		dx_vals = diff(x)
		dx_mean = mean(dx_vals)
		dx_std = std(dx_vals)

		if dx_std / dx_mean > 0.05
			@warn "Savitzky-Golay: Data is not uniformly spaced (std/mean = $(dx_std/dx_mean)). Results may be inaccurate."
		end

		# Use mean spacing for derivative scaling
		dx = dx_mean

		# Compute derivatives for each order
		for order in orders
			try
				# Check if this order is feasible
				if polyorder < order
					predictions[order] = fill(NaN, length(x_eval))
					failures[order] = "Polynomial order ($polyorder) < derivative order ($order)"
					continue
				end

				# Apply SG filter to get derivatives at original x points
				deriv_at_x = apply_savitzky_golay_filter(y, window, polyorder, order, dx)

				# Find valid (non-NaN) points for interpolation
				valid_mask = .!isnan.(deriv_at_x)

				if sum(valid_mask) < 2
					predictions[order] = fill(NaN, length(x_eval))
					failures[order] = "Not enough valid points after filtering (boundaries removed)"
					continue
				end

				# Interpolate derivative values to x_eval points
				# Use simple linear interpolation (fast and robust)
				x_valid = x[valid_mask]
				deriv_valid = deriv_at_x[valid_mask]

				# Interpolate to x_eval
				deriv_eval = similar(x_eval)
				for (i, xi) in enumerate(x_eval)
					if xi < x_valid[1] || xi > x_valid[end]
						# Extrapolation: use nearest value (flat extrapolation)
						if xi < x_valid[1]
							deriv_eval[i] = deriv_valid[1]
						else
							deriv_eval[i] = deriv_valid[end]
						end
					else
						# Linear interpolation
						idx = searchsortedfirst(x_valid, xi)
						if idx == 1
							deriv_eval[i] = deriv_valid[1]
						elseif idx > length(x_valid)
							deriv_eval[i] = deriv_valid[end]
						else
							t = (xi - x_valid[idx-1]) / (x_valid[idx] - x_valid[idx-1])
							deriv_eval[i] = (1 - t) * deriv_valid[idx-1] + t * deriv_valid[idx]
						end
					end
				end

				predictions[order] = deriv_eval

			catch e
				predictions[order] = fill(NaN, length(x_eval))
				failures[order] = string(e)
			end
		end

		timing = time() - t_start
		return MethodResult("Savitzky-Golay", "Local Polynomial", predictions, failures, timing, true)

	catch e
		@warn "Savitzky-Golay failed" exception = e
		timing = time() - t_start
		return MethodResult("Savitzky-Golay", "Error", Dict(), Dict(0 => string(e)), timing, false)
	end
end
