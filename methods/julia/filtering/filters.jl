"""
Filtering Methods for Derivative Estimation

Implements Savitzky-Golay filter for smoothing and differentiation.
"""

# Load common utilities
include("../common.jl")

using LinearAlgebra

# Import hyperparameter selection for noise estimation
include("../../../src/hyperparameter_selection.jl")
using .HyperparameterSelection

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
	evaluate_savitzky_golay_fixed(x, y, x_eval, orders; params=Dict())

Evaluate Savitzky-Golay method with FIXED window and polynomial order.

This version uses fixed parameters optimized for N≈101, orders 0-7:
- Window: 15 (balanced for N=101)
- Polyorder: 7 (supports up to 7th derivative)

# Arguments
- `x`: Training x points (assumed approximately uniformly spaced)
- `y`: Training y values
- `x_eval`: Evaluation points
- `orders`: Derivative orders to compute
- `params`: Optional parameters
  - `:sg_window`: Window size (default: 15, must be odd)
  - `:sg_polyorder`: Polynomial order (default: 7)

# Returns
MethodResult with predictions for each derivative order
"""
function evaluate_savitzky_golay_fixed(
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
		# Fixed parameters optimized for N≈101, orders 0-7
		window = get(params, :sg_window, 15)  # Smaller than 21, better for N=101
		window = isodd(window) ? window : window + 1  # Ensure odd
		window = min(window, length(x))
		polyorder = get(params, :sg_polyorder, 7)  # Higher than 5, supports order 7

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
		return MethodResult("SavitzkyGolay-Fixed", "Local Polynomial", predictions, failures, timing, true)

	catch e
		@warn "Savitzky-Golay-Fixed failed" exception = e
		timing = time() - t_start
		return MethodResult("SavitzkyGolay-Fixed", "Error", Dict(), Dict(0 => string(e)), timing, false)
	end
end


"""
	evaluate_savitzky_golay_adaptive(x, y, x_eval, orders; params=Dict())

Evaluate Savitzky-Golay method with ADAPTIVE noise-dependent window sizing.

This addresses the bias floors observed in fixed-window S-G by adapting the window
size based on estimated noise level and signal roughness. Uses larger windows for
higher derivative orders to combat variance growth.

Key features:
- Noise-adaptive window sizing using MAD estimator
- Polynomial order tiers: r≤3→p=7; 4-5→p=9; 6-7→p=11
- Per-order optimization (w_r, p_r) for each derivative
- Window scaling: h* ∝ (σ²/ρ²)^{1/(2p+3)} from MISE minimization

# Arguments
- `x`: Training x points (assumed approximately uniformly spaced)
- `y`: Training y values
- `x_eval`: Evaluation points
- `orders`: Derivative orders to compute
- `params`: Optional parameters (reserved for future use)

# Returns
MethodResult with predictions for each derivative order
"""
function evaluate_savitzky_golay_adaptive(
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
		n = length(x)

		# Check for uniform spacing
		dx_vals = diff(x)
		dx_mean = mean(dx_vals)
		dx_std = std(dx_vals)

		if dx_std / dx_mean > 0.05
			@warn "Savitzky-Golay-Adaptive: Data is not uniformly spaced (std/mean = $(dx_std/dx_mean))"
		end

		dx = dx_mean

		# Robust noise estimation using wavelet MAD (Donoho-Johnstone 1994)
		# The "gold standard" noise estimator - uses finest-scale wavelet detail coefficients
		# Far more accurate than finite-difference methods for smooth signals (40-7600× better)
		σ_hat = HyperparameterSelection.estimate_noise_wavelet(y)

		# Roughness estimation using 4th-order differences (proxy for signal smoothness)
		# Normalized by dx^4 for scale consistency
		if n >= 5
			# Apply diff 4 times to get 4th-order differences
			d4 = y
			for _ in 1:4
				d4 = diff(d4)
			end
			ρ_hat = sqrt(mean(d4 .^ 2)) / (dx^4 + 1e-24)
		else
			ρ_hat = 1.0
		end

		# Calibration constants c_{p,r} - empirically tuned for N≈100
		# These balance bias-variance tradeoff for different (polyorder, deriv_order) combinations
		c_pr = Dict(
			(7, 0) => 1.0, (7, 1) => 1.1, (7, 2) => 1.2, (7, 3) => 1.3,
			(9, 0) => 1.0, (9, 1) => 1.1, (9, 2) => 1.2, (9, 3) => 1.3, (9, 4) => 1.4, (9, 5) => 1.5,
			(11, 0) => 1.0, (11, 1) => 1.1, (11, 2) => 1.2, (11, 3) => 1.3, (11, 4) => 1.4,
			(11, 5) => 1.5, (11, 6) => 1.6, (11, 7) => 1.7,
		)

		# Window cap for safety: N/3 for N=101 gives max window of 33
		max_window = max(5, div(n, 3))
		if iseven(max_window)
			max_window -= 1
		end

		# Compute derivatives for each order with adaptive parameters
		for order in orders
			try
				# Determine polynomial order based on derivative order (tiered approach)
				p = if order <= 3
					7
				elseif order <= 5
					9
				else
					11
				end

				# Get calibration constant
				c = get(c_pr, (p, order), 1.0 + 0.1 * order)

				# Compute optimal window via plug-in rule
				# h* ∝ (σ²/ρ²)^{1/(2p+3)}
				ratio = max(σ_hat, 1e-24)^2 / max(ρ_hat, 1e-24)^2
				h_star = c * (ratio^(1.0 / (2 * p + 3)))

				# Convert to window length in samples
				w_ideal = Int(2 * floor(h_star / max(dx, 1e-24)) + 1)

				# Apply constraints
				w = max(w_ideal, p + 3)  # Need at least p+3 for numerical headroom
				w = min(w, max_window)   # Safety cap
				w = min(w, n)            # Can't exceed data size

				# Ensure odd
				if iseven(w)
					w -= 1
				end

				# Final check: window must be > polyorder
				if w <= p
					w = p + 2
					if iseven(w)
						w += 1
					end
					w = min(w, n)
					if iseven(w)
						w -= 1
					end
				end

				# Check feasibility
				if p < order
					predictions[order] = fill(NaN, length(x_eval))
					failures[order] = "Polynomial order ($p) < derivative order ($order)"
					continue
				end

				if w > n
					predictions[order] = fill(NaN, length(x_eval))
					failures[order] = "Window ($w) > data size ($n)"
					continue
				end

				# Apply SG filter to get derivatives at original x points
				deriv_at_x = apply_savitzky_golay_filter(y, w, p, order, dx)

				# Find valid (non-NaN) points for interpolation
				valid_mask = .!isnan.(deriv_at_x)

				if sum(valid_mask) < 2
					predictions[order] = fill(NaN, length(x_eval))
					failures[order] = "Not enough valid points (window=$w, poly=$p)"
					continue
				end

				# Interpolate derivative values to x_eval points
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
		return MethodResult("SavitzkyGolay-Adaptive", "Local Polynomial", predictions, failures, timing, true)

	catch e
		@warn "Savitzky-Golay-Adaptive failed" exception = e
		timing = time() - t_start
		return MethodResult("SavitzkyGolay-Adaptive", "Error", Dict(), Dict(0 => string(e)), timing, false)
	end
end
