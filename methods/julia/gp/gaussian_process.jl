"""
Gaussian Process Methods for Derivative Estimation

Implements 5 GP-based methods:
1. GP-Julia-SE: Analytic SE (RBF) kernel with closed-form derivatives
2. GP-Julia-AD: AD-based GP with generic kernels
3. GP-Julia-Matern-0.5: Matérn-1/2 kernel (exponential)
4. GP-Julia-Matern-1.5: Matérn-3/2 kernel (once differentiable)
5. GP-Julia-Matern-2.5: Matérn-5/2 kernel (twice differentiable)
"""

# Load common utilities
include("../common.jl")

using GaussianProcesses
using Optim
using LineSearches
using Dates  # For timestamp logging
# Note: Suppressor removed - @suppress macro causes threading issues (file descriptor conflicts)

# ============================================================================
# GP-Julia-SE: Analytic SE (RBF) GP with closed-form derivatives
# ============================================================================

"""
	fit_gp_se_analytic(x, y)

Fit a 1D SE GP by MLE and return a callable that yields n-th derivatives using
closed-form kernel derivatives (probabilists' Hermite polynomials).
"""
function fit_gp_se_analytic(x::Vector{Float64}, y::Vector{Float64})
	n = length(x)
	@assert n == length(y)

	# Center y to handle mean; derivatives of constant are zero
	y_mean = mean(y)
	yc = y .- y_mean

	# Normalize x for better conditioning
	x_mean = mean(x)
	x_std = std(x)
	x_std = x_std == 0.0 ? 1.0 : x_std
	x_scaled = (x .- x_mean) ./ x_std

	# Precompute pairwise squared distances matrix (for efficiency)
	function sqdist_matrix(xv::Vector{Float64})
		X = repeat(xv, 1, length(xv))
		return (X .- X') .^ 2
	end
	D2 = sqdist_matrix(x_scaled)

	# Robust Cholesky with escalating jitter (used both in NLL and final)
	function cholesky_with_jitter!(K::AbstractMatrix{T}) where T <: Real
		d = diag(K)
		# Use primal values to avoid ordering on Dual numbers
		dpr = map(x -> ForwardDiff.value(x), d)
		dmed = median(dpr)
		base = max(T(1e-12), T(1e-8) * T(dmed))
		jitter = base
		for _ in 1:12
			try
				return cholesky(Symmetric(K))
			catch
				@inbounds for i in 1:size(K, 1)
					K[i, i] += jitter
				end
				jitter *= T(10)
			end
		end
		return cholesky(Symmetric(K); check = false)
	end

	# Negative log marginal likelihood (with log-params) using robust PD handling
	function nll(p)
		ℓ = exp(p[1])
		σf = exp(p[2])
		σn = max(exp(p[3]), 1e-6)
		K = (σf^2) .* exp.(-0.5 .* (D2 ./ (ℓ^2)))
		@inbounds for i in 1:n
			K[i, i] += σn^2
		end
		# tiny base jitter before attempting factorization
		@inbounds for i in 1:n
			K[i, i] += 1e-12
		end
		F = cholesky_with_jitter!(K)
		α = F \ yc
		return 0.5 * dot(yc, α) + sum(log, diag(F.U)) + 0.5 * n * log(2π)
	end

	# Initial params from heuristics
	p0 = [log(std(x) / 8), log(std(yc) + eps()), log(std(yc) / 100 + 1e-6)]
	# Use finite-difference gradients to avoid Dual types inside Cholesky
	result = Optim.optimize(nll, p0, LBFGS(); autodiff = :finite)
	p̂ = Optim.minimizer(result)
	ℓ̂ = exp(p̂[1])
	σf̂ = exp(p̂[2])
	σn̂ = exp(p̂[3])

	# Precompute training structures
	K̂ = (σf̂^2) .* exp.(-0.5 .* (D2 ./ (ℓ̂^2)))
	@inbounds for i in 1:n
		K̂[i, i] += σn̂^2
	end
	F̂ = cholesky_with_jitter!(K̂)
	α̂ = F̂ \ yc

	# Probabilists' Hermite polynomials He_n(u)
	# FIX: Clamping u to prevent overflow in recursion
	function hermite_prob(n::Int, u::AbstractVector{T}) where {T}
		# Clamp input to prevent overflow (exp(-0.5*u^2) becomes negligible beyond |u|=50)
		u_safe = clamp.(u, T(-50), T(50))

		if n == 0
			return ones(T, length(u_safe))
		elseif n == 1
			return copy(u_safe)
		else
			He_nm1 = ones(T, length(u_safe))
			He_n0 = copy(u_safe)
			for k in 1:(n-1)
				He_np1 = u_safe .* He_n0 .- k .* He_nm1
				He_nm1, He_n0 = He_n0, He_np1
			end
			return He_n0
		end
	end

	# Vectorized evaluator for n-th derivative of posterior mean at x*
	function eval_nth_deriv(xstar::Float64, n::Int)
		if n == 0
			# Standard predictive mean: k_*^T α̂ + y_mean
			xstar_scaled = (xstar - x_mean) / x_std
			u = (xstar_scaled .- x_scaled) ./ ℓ̂
			kstar = (σf̂^2) .* exp.(-0.5 .* (u .^ 2))
			return dot(kstar, α̂) + y_mean
		else
			xstar_scaled = (xstar - x_mean) / x_std
			u = (xstar_scaled .- x_scaled) ./ ℓ̂

			# Check for numerical instability before computing
			if maximum(abs.(u)) > 10.0
				@warn "GP-Julia-SE: Large u values (max=$(maximum(abs.(u)))), order $n may be unstable"
			end

			base = exp.(-0.5 .* (u .^ 2))
			He = hermite_prob(n, u)
			sign = (n % 2 == 1) ? -1.0 : 1.0
			# Clamp scaling factor to prevent overflow
			scale_raw = (σf̂^2) * (ℓ̂ ^ (-n))
			scale = clamp(scale_raw, -1e10, 1e10)

			k_n = (sign .* scale) .* He .* base
			result = dot(k_n, α̂)

			# Return NaN if result is non-finite
			return isfinite(result) ? result : NaN
		end
	end

	return (z::Float64, n::Int = 0) -> eval_nth_deriv(z, n)
end


# ============================================================================
# GP-TaylorAD-Julia: AD-based GP with generic kernels
# ============================================================================

"""
	fit_gp(x, y; kernel=:SE, optimize=true)

Fit Gaussian Process with specified kernel.
"""
function fit_gp(x, y; kernel = :SE, optimize = true)
	# Scale data for numerical stability
	x_mean, x_std = mean(x), std(x)
	y_mean, y_std = mean(y), std(y)
	x_scaled = (x .- x_mean) ./ x_std
	y_scaled = (y .- y_mean) ./ max(y_std, 1e-8)

	# Optional tiny jitter to stabilize factorization
	y_jittered = y_scaled .+ 1e-8 .* randn(length(y_scaled))

	# Choose kernel (SE/RBF)
	kern = SEIso(0.0, 0.0)

	# Create GP with a modest noise floor
	gp = GP(x_scaled, y_jittered, MeanZero(), kern, log(1e-2))

	# Optimize hyperparameters (robust to failures)
	if optimize
		try
			optimize!(gp; method = LBFGS(linesearch = LineSearches.BackTracking()))
		catch e
			@warn "GP optimization failed, proceeding with current hyperparameters" exception=(e,)
		end
	end

	# AD-friendly predictor: use predict_f, not predict_y
	gp_func = z -> begin
		z_scaled = (z - x_mean) / x_std
		pred, _ = predict_f(gp, [z_scaled])
		return pred[1] * y_std + y_mean
	end

	return gp_func
end

"""
	fit_gp_ad(x, y; rng=Random.GLOBAL_RNG)

Fit GP using TaylorDiff-based automatic differentiation (simpler, more robust).
Based on ODEParameterEstimation approach - normalize, optimize, then AD through predictions.

Args:
	x: Input locations
	y: Observations
	rng: Random number generator for reproducible jitter (default: global RNG)
"""
function fit_gp_ad(x::Vector{Float64}, y::Vector{Float64}; rng = Random.GLOBAL_RNG)
	@assert length(x) == length(y) "Input arrays must have same length"

	# 1. Normalize y values
	y_mean = mean(y)
	y_std = std(y)
	y_normalized = (y .- y_mean) ./ max(y_std, 1e-8)  # Avoid division by very small numbers

	# Configure initial GP hyperparameters
	initial_lengthscale = log(std(x) / 8)
	initial_variance = 0.0
	initial_noise = -2.0

	# Add small amount of jitter to avoid numerical issues
	# IMPORTANT: Use provided RNG for reproducibility
	kernel = SEIso(initial_lengthscale, initial_variance)
	jitter = 1e-8
	y_jittered = y_normalized .+ jitter * randn(rng, length(y_normalized))

	# 2. Do GPR approximation on normalized data
	# Note: @suppress removed due to threading issues (file descriptor conflicts in parallel execution)
	local gp
	gp = GP(x, y_jittered, MeanZero(), kernel, initial_noise)
	GaussianProcesses.optimize!(gp; method = LBFGS(linesearch = LineSearches.BackTracking()))

	# Create a function that evaluates the GPR prediction and denormalizes the output
	function denormalized_gpr(x_eval)
		pred, _ = predict_f(gp, [x_eval])
		return y_std * pred[1] + y_mean
	end

	return denormalized_gpr
end


# ============================================================================
# GP-Julia-Matern: GP with Matérn kernels
# ============================================================================

"""
	fit_gp_matern(x, y; nu=1.5, rng=Random.GLOBAL_RNG)

Fit GP with Matérn kernel using TaylorDiff-based automatic differentiation.
Uses same approach as GP-Julia-AD but with Matérn kernel instead of SE.

Supported nu values: 0.5 (Mat12Iso), 1.5 (Mat32Iso), 2.5 (Mat52Iso)

Args:
	x: Input locations
	y: Observations
	nu: Matérn smoothness parameter
	rng: Random number generator for reproducible jitter (default: global RNG)
"""
function fit_gp_matern(x::Vector{Float64}, y::Vector{Float64}; nu::Float64 = 1.5, rng = Random.GLOBAL_RNG)
	@assert length(x) == length(y) "Input arrays must have same length"

	# 1. Normalize y values
	y_mean = mean(y)
	y_std = std(y)
	y_normalized = (y .- y_mean) ./ max(y_std, 1e-8)

	# Configure initial GP hyperparameters
	initial_lengthscale = log(std(x) / 8)
	initial_variance = 0.0
	initial_noise = -2.0

	# Select Matérn kernel based on nu
	kernel = if abs(nu - 0.5) < 1e-8
		Mat12Iso(initial_lengthscale, initial_variance)  # Matérn-1/2
	elseif abs(nu - 1.5) < 1e-8
		Mat32Iso(initial_lengthscale, initial_variance)  # Matérn-3/2
	elseif abs(nu - 2.5) < 1e-8
		Mat52Iso(initial_lengthscale, initial_variance)  # Matérn-5/2
	else
		error("Unsupported nu value: $nu (use 0.5, 1.5, or 2.5)")
	end

	# Add small amount of jitter to avoid numerical issues
	# IMPORTANT: Use provided RNG for reproducibility
	jitter = 1e-8
	y_jittered = y_normalized .+ jitter * randn(rng, length(y_normalized))

	# 2. Do GPR approximation on normalized data
	# Note: @suppress removed due to threading issues (file descriptor conflicts in parallel execution)
	local gp
	gp = GP(x, y_jittered, MeanZero(), kernel, initial_noise)
	GaussianProcesses.optimize!(gp; method = LBFGS(linesearch = LineSearches.BackTracking()))

	# Create a function that evaluates the GPR prediction and denormalizes the output
	function denormalized_gpr(x_eval)
		pred, _ = predict_f(gp, [x_eval])
		return y_std * pred[1] + y_mean
	end

	return denormalized_gpr
end


# ============================================================================
# Method Evaluators (Standard API)
# ============================================================================

"""
	evaluate_gp_se(x, y, x_eval, orders; params=Dict())

Evaluate GP-Julia-SE method: Analytic SE kernel with closed-form derivatives.
"""
function evaluate_gp_se(
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
		# Fit analytic SE GP
		gp_analytic = fit_gp_se_analytic(x, y)

		# Evaluate derivatives directly from analytic SE GP
		for order in orders
			try
				predictions[order] = [gp_analytic(xi, order) for xi in x_eval]
			catch e
				predictions[order] = fill(NaN, length(x_eval))
				failures[order] = string(e)
			end
		end

		timing = time() - t_start
		return MethodResult("GP-Julia-SE", "Gaussian Process", predictions, failures, timing, true)

	catch e
		@warn "GP-Julia-SE failed" exception=e
		timing = time() - t_start
		return MethodResult("GP-Julia-SE", "Error", Dict(), Dict(0 => string(e)), timing, false)
	end
end

"""
	evaluate_gp_ad(x, y, x_eval, orders; params=Dict())

Evaluate GP-Julia-AD method: AD-based GP with generic kernels.
"""
function evaluate_gp_ad(
	x::Vector{Float64},
	y::Vector{Float64},
	x_eval::Vector{Float64},
	orders::Vector{Int};
	params = Dict()
)
	t_start = time()
	predictions = Dict{Int, Vector{Float64}}()
	failures = Dict{Int, String}()

	# Setup verbose logging to gitignored file
	trial_id = get(params, :trial_id, "unknown")
	log_dir = joinpath(dirname(@__DIR__), "..", "..", "build", "logs", "gp")
	mkpath(log_dir)
	log_file = joinpath(log_dir, "$(trial_id).log")

	# Extract RNG from params for reproducibility
	rng = get(params, :rng, Random.GLOBAL_RNG)
	rng_seed = get(params, :rng_seed, "unknown")

	# Log trial start
	open(log_file, "w") do io
		println(io, "=" ^ 80)
		println(io, "GP-TaylorAD-Julia Detailed Log")
		println(io, "=" ^ 80)
		println(io, "Trial ID: $trial_id")
		println(io, "Timestamp: $(now())")
		println(io, "RNG Seed: $rng_seed")
		println(io)
		println(io, "Data Statistics:")
		println(io, "  Number of points: $(length(x))")
		println(io, "  x range: [$(minimum(x)), $(maximum(x))]")
		println(io, "  y range: [$(minimum(y)), $(maximum(y))]")
		println(io, "  y mean: $(mean(y))")
		println(io, "  y std: $(std(y))")
		println(io, "  Orders requested: $orders")
		println(io)
	end

	try
		# Fit AD-based GP
		fitted_func = fit_gp_ad(x, y; rng = rng)

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

		# Log success
		open(log_file, "a") do io
			println(io, "Status: SUCCESS")
			println(io, "Total time: $(timing) seconds")
			println(io, "Orders computed: $(sort(collect(keys(predictions))))")
			if !isempty(failures)
				println(io, "Partial failures: $(sort(collect(keys(failures))))")
			end
			println(io, "=" ^ 80)
		end

		return MethodResult("GP-TaylorAD-Julia", "Gaussian Process", predictions, failures, timing, true)

	catch e
		# Enhanced error logging for debugging GP failures
		error_details = sprint(showerror, e, catch_backtrace())
		timing = time() - t_start

		# Log failure to file
		open(log_file, "a") do io
			println(io, "Status: FAILED")
			println(io, "Total time: $(timing) seconds")
			println(io)
			println(io, "Error Details:")
			println(io, error_details)
			println(io, "=" ^ 80)
		end

		# Also log to console
		@error "GP-TaylorAD-Julia FAILED" exception=(e, catch_backtrace()) data_stats=(
			n_points=length(x),
			x_range=(minimum(x), maximum(x)),
			y_range=(minimum(y), maximum(y)),
			y_std=std(y),
			y_mean=mean(y)
		)

		return MethodResult("GP-TaylorAD-Julia", "Error", Dict(), Dict(0 => error_details), timing, false)
	end
end

"""
	evaluate_gp_matern(x, y, x_eval, orders; nu=1.5, params=Dict())

Evaluate GP-Julia-Matern methods with specified nu parameter.
"""
function evaluate_gp_matern(
	x::Vector{Float64},
	y::Vector{Float64},
	x_eval::Vector{Float64},
	orders::Vector{Int};
	nu::Float64 = 1.5,
	params = Dict()
)
	t_start = time()
	predictions = Dict{Int, Vector{Float64}}()
	failures = Dict{Int, String}()

	# Determine method name based on nu
	method_name = "GP-Julia-Matern-$(nu)"

	try
		# Fit Matérn GP
		# Extract RNG from params for reproducibility
		rng = get(params, :rng, Random.GLOBAL_RNG)
		fitted_func = fit_gp_matern(x, y; nu = nu, rng = rng)

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
		return MethodResult(method_name, "Gaussian Process", predictions, failures, timing, true)

	catch e
		@warn "$method_name failed" exception=e
		timing = time() - t_start
		return MethodResult(method_name, "Error", Dict(), Dict(0 => string(e)), timing, false)
	end
end

# Convenience wrappers for specific nu values
evaluate_gp_matern_05(x, y, x_eval, orders; params = Dict()) =
	evaluate_gp_matern(x, y, x_eval, orders; nu = 0.5, params = params)

evaluate_gp_matern_15(x, y, x_eval, orders; params = Dict()) =
	evaluate_gp_matern(x, y, x_eval, orders; nu = 1.5, params = params)

evaluate_gp_matern_25(x, y, x_eval, orders; params = Dict()) =
	evaluate_gp_matern(x, y, x_eval, orders; nu = 2.5, params = params)
