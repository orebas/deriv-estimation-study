"""
Regularization Methods for Derivative Estimation

Implements:
1. TrendFilter-k7: Trend filtering with order-7 penalty
2. TrendFilter-k2: Trend filtering with order-2 penalty
3. TVRegDiff-Julia: Total variation regularized differentiation (orders 0-1 only)
"""

# Load common utilities
include("../common.jl")

using Lasso
using NoiseRobustDifferentiation

# ============================================================================
# Trend Filtering
# ============================================================================

"""
	fit_trend_filter(x, y; order=7, λ=0.1)

Trend filtering using Lasso.jl.
"""
function fit_trend_filter(x, y; order = 7, λ = 0.1)
	# Fit trend filter
	# Note: TrendFilter in Lasso.jl penalizes the (order+1)-th derivative
	tf = fit(TrendFilter, y, order, λ)

	# Extract fitted values directly from the coefficient field
	y_fitted = tf.β

	# Return interpolator (linear interpolation of fitted values)
	return z -> begin
		# Simple linear interpolation
		idx = searchsortedfirst(x, z)
		if idx == 1
			return y_fitted[1]
		elseif idx > length(x)
			return y_fitted[end]
		else
			# Linear interpolation
			t = (z - x[idx-1]) / (x[idx] - x[idx-1])
			return (1 - t) * y_fitted[idx-1] + t * y_fitted[idx]
		end
	end
end


# ============================================================================
# Method Evaluators (Standard API)
# ============================================================================

"""
	evaluate_trend_filter(x, y, x_eval, orders; order=7, params=Dict())

Evaluate TrendFilter method with specified penalty order.
"""
function evaluate_trend_filter(
	x::Vector{Float64},
	y::Vector{Float64},
	x_eval::Vector{Float64},
	orders::Vector{Int};
	filter_order::Int = 7,
	params = Dict()
)
	t_start = time()
	predictions = Dict{Int, Vector{Float64}}()
	failures = Dict{Int, String}()

	# Determine method name
	method_name = "TrendFilter-k$(filter_order)"

	try
		# Get lambda parameter
		λ = get(params, :tf_lambda, 0.1)

		# Fit trend filter
		fitted_func = fit_trend_filter(x, y; order = filter_order, λ = λ)

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
		return MethodResult(method_name, "Regularization", predictions, failures, timing, true)

	catch e
		@warn "$method_name failed" exception=e
		timing = time() - t_start
		return MethodResult(method_name, "Error", Dict(), Dict(0 => string(e)), timing, false)
	end
end

# Convenience wrappers for specific orders
evaluate_trend_filter_k7(x, y, x_eval, orders; params = Dict()) =
	evaluate_trend_filter(x, y, x_eval, orders; filter_order = 7, params = params)

evaluate_trend_filter_k2(x, y, x_eval, orders; params = Dict()) =
	evaluate_trend_filter(x, y, x_eval, orders; filter_order = 2, params = params)

"""
	evaluate_tvregdiff(x, y, x_eval, orders; params=Dict())

Evaluate TVRegDiff-Julia method: Total variation regularized differentiation.
NOTE: Limited to orders 0-1 only (iterative approach numerically unstable for higher orders).
"""
function evaluate_tvregdiff(
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
		iters = get(params, :tv_iters, 100)
		alpha = get(params, :tv_alpha, 1e-2)
		scale = get(params, :tv_scale, "small")
		diff_kernel = get(params, :tv_diff_kernel, "abs")  # "abs" or "square"

		# Ensure keyword values have expected types for NoiseRobustDifferentiation
		scale_str = scale isa AbstractString ? String(scale) : String(Symbol(scale))
		diffkernel_str = diff_kernel isa AbstractString ? String(diff_kernel) : String(Symbol(diff_kernel))

		# Grid spacing
		dt = mean(diff(x))

		# Store derivatives - only compute order 0 and 1
		tv_derivatives = Dict{Int, Vector{Float64}}()
		tv_derivatives[0] = y  # Order 0 is the original data

		# Only compute 1st derivative (iterative approach fails for higher orders)
		if any(o -> o >= 1, orders)
			deriv1 = NoiseRobustDifferentiation.tvdiff(
				y, iters, alpha;
				dx = dt, scale = scale_str,
				show_diagn = false, diff_kernel = diffkernel_str,
				cg_tol = 1e-6, cg_maxiter = 100
			)
			tv_derivatives[1] = deriv1
		end

		# For evaluation at arbitrary points, use simple linear interpolation
		# (TV-regularization already denoised, so simple interpolation suffices)
		for order in orders
			try
				if haskey(tv_derivatives, order)
					deriv_data = tv_derivatives[order]
					# Linear interpolation for each evaluation point
					predictions[order] = map(x_eval) do xi
						idx = searchsortedfirst(x, xi)
						if idx == 1
							deriv_data[1]
						elseif idx > length(x)
							deriv_data[end]
						else
							t = (xi - x[idx-1]) / (x[idx] - x[idx-1])
							(1 - t) * deriv_data[idx-1] + t * deriv_data[idx]
						end
					end
				else
					# Orders > 1 not supported (iterative approach numerically unstable)
					predictions[order] = fill(NaN, length(x_eval))
					failures[order] = "TVRegDiff limited to orders 0-1 (iterative differentiation unstable for higher orders)"
				end
			catch e
				predictions[order] = fill(NaN, length(x_eval))
				failures[order] = string(e)
			end
		end

		timing = time() - t_start
		return MethodResult("TVRegDiff-Julia", "Regularization", predictions, failures, timing, true)

	catch e
		@warn "TVRegDiff-Julia failed" exception=e
		timing = time() - t_start
		return MethodResult("TVRegDiff-Julia", "Error", Dict(), Dict(0 => string(e)), timing, false)
	end
end
