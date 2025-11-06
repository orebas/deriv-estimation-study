"""
AAA Rational Approximation Methods for Derivative Estimation

Implements 4 AAA-based methods with different tolerance strategies:
1. AAA-HighPrec: Fixed high-precision tolerance (1e-14)
2. AAA-LowPrec: Fixed or adaptive tolerance (0.1 or noise-based)
3. AAA-Adaptive-Diff2: Adaptive tolerance using 2nd-order difference noise estimation
4. AAA-Adaptive-Wavelet: Adaptive tolerance using Haar wavelet noise estimation

All methods use barycentric rational approximation via the AAA algorithm and compute
derivatives using TaylorDiff automatic differentiation.
"""

# Load common utilities
include("../common.jl")

# ============================================================================
# Method Evaluators (Standard API)
# ============================================================================

"""
	evaluate_aaa_highprec(x, y, x_eval, orders; params=Dict())

Evaluate AAA-HighPrec method: Fixed high-precision tolerance (1e-14).
"""
function evaluate_aaa_highprec(
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
		# Fit AAA with high precision
		fitted_func = fit_aaa(x, y; tol = 1e-14)

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
		return MethodResult("AAA-HighPrec", "Rational", predictions, failures, timing, true)

	catch e
		@warn "AAA-HighPrec failed" exception=e
		timing = time() - t_start
		return MethodResult("AAA-HighPrec", "Error", Dict(), Dict(0 => string(e)), timing, false)
	end
end

"""
	evaluate_aaa_lowprec(x, y, x_eval, orders; params=Dict())

Evaluate AAA-LowPrec method: Fixed tolerance (0.1) or adaptive based on environment variable.
"""
function evaluate_aaa_lowprec(
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
		# Check if we should use adaptive tolerance
		use_adaptive = get(ENV, "USE_ADAPTIVE_AAA", "0") != "0"

		if use_adaptive
			tol_adaptive = HyperparameterSelection.select_aaa_tolerance(y)
			fitted_func = fit_aaa(x, y; tol = tol_adaptive)
		else
			fitted_func = fit_aaa(x, y; tol = 0.1)
		end

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
		return MethodResult("AAA-LowPrec", "Rational", predictions, failures, timing, true)

	catch e
		@warn "AAA-LowPrec failed" exception=e
		timing = time() - t_start
		return MethodResult("AAA-LowPrec", "Error", Dict(), Dict(0 => string(e)), timing, false)
	end
end

"""
	evaluate_aaa_adaptive_diff2(x, y, x_eval, orders; params=Dict())

Evaluate AAA-Adaptive-Diff2 method: Adaptive tolerance using 2nd-order difference noise estimation.
"""
function evaluate_aaa_adaptive_diff2(
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
		# Adaptive tolerance using 2nd-order difference noise estimation
		σ̂ = HyperparameterSelection.estimate_noise_diff2(y)
		tol_adaptive = max(1e-13, 10.0 * σ̂)
		fitted_func = fit_aaa(x, y; tol = tol_adaptive)

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
		return MethodResult("AAA-Adaptive-Diff2", "Rational", predictions, failures, timing, true)

	catch e
		@warn "AAA-Adaptive-Diff2 failed" exception=e
		timing = time() - t_start
		return MethodResult("AAA-Adaptive-Diff2", "Error", Dict(), Dict(0 => string(e)), timing, false)
	end
end

"""
	evaluate_aaa_adaptive_wavelet(x, y, x_eval, orders; params=Dict())

Evaluate AAA-Adaptive-Wavelet method: Adaptive tolerance using Haar wavelet noise estimation.
"""
function evaluate_aaa_adaptive_wavelet(
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
		# Adaptive tolerance using Haar wavelet noise estimation
		σ̂ = HyperparameterSelection.estimate_noise_wavelet(y)
		tol_adaptive = max(1e-13, 10.0 * σ̂)
		fitted_func = fit_aaa(x, y; tol = tol_adaptive)

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
		return MethodResult("AAA-Adaptive-Wavelet", "Rational", predictions, failures, timing, true)

	catch e
		@warn "AAA-Adaptive-Wavelet failed" exception=e
		timing = time() - t_start
		return MethodResult("AAA-Adaptive-Wavelet", "Error", Dict(), Dict(0 => string(e)), timing, false)
	end
end
