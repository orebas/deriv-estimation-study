"""
Julia-based Differentiation Methods

Implements all Julia-native methods for high-order derivative estimation.
"""

using BaryRational
using GaussianProcesses
using Optim
using LineSearches
using Suppressor
using ForwardDiff
using TaylorDiff
using Dierckx
using Lasso
using FFTW
using Statistics
using LinearAlgebra
using NoiseRobustDifferentiation

# ============================================================================
# Method Result Structure
# ============================================================================

"""Result from a single method evaluation"""
struct MethodResult
	name::String
	category::String
	predictions::Dict{Int, Vector{Float64}}  # order => values
	failures::Dict{Int, String}  # order => error message
	timing::Float64
	success::Bool
end


# ============================================================================
# Derivative Computation Utilities
# ============================================================================

"""
	nth_deriv_at(f, n, t)

Compute nth derivative of function f at point t using ForwardDiff (recursive).
"""
function nth_deriv_at(f, n::Int, t)
	n == 0 && return f(t)
	n == 1 && return ForwardDiff.derivative(f, t)
	g(t) = nth_deriv_at(f, n - 1, t)
	return ForwardDiff.derivative(g, t)
end

"""
	nth_deriv_taylor(f, n, t)

Compute nth derivative using TaylorDiff (faster for n > 1).
"""
function nth_deriv_taylor(f, n::Int, t)
	n == 0 && return f(t)
	return TaylorDiff.derivative(f, t, Val(n))
end

"""
	compute_derivatives_at_points(f, x_eval, orders; method=:taylor)

Compute derivatives of fitted function f at evaluation points.
"""
function compute_derivatives_at_points(f, x_eval, orders; method = :taylor)
	results = Dict{Int, Vector{Float64}}()

	for order in orders
		try
			if method == :taylor && order > 0
				results[order] = [nth_deriv_taylor(f, order, x) for x in x_eval]
			else
				results[order] = [nth_deriv_at(f, order, x) for x in x_eval]
			end
		catch e
			# Return NaNs if differentiation fails
			results[order] = fill(NaN, length(x_eval))
		end
	end

	return results
end


# ============================================================================
# AAA Rational Approximation (from bary_derivs.jl)
# ============================================================================

"""Barycentric evaluation"""
function bary_eval(z, f::Vector{T}, x::Vector{T}, w::Vector{T}, tol = 1e-13) where {T}
	num, den = zero(T), zero(T)
	breakflag, breakindex = false, -1

	for j in eachindex(f)
		if abs(z - x[j]) < sqrt(tol)
			breakflag = true
			breakindex = j
			break
		end
		t = w[j] / (z - x[j])
		num += t * f[j]
		den += t
	end

	if breakflag
		# Near a support point, use special formula
		num, den = zero(T), zero(T)
		for j in eachindex(f)
			if j != breakindex
				t = w[j] / (z - x[j])
				num += t * f[j]
				den += t
			end
		end
		m = z - x[breakindex]
		return (w[breakindex] * f[breakindex] + m * num) / (w[breakindex] + m * den)
	end

	return num / den
end

"""Wrapper for AAA approximation"""
struct AAAApprox
	f::Vector{Float64}
	x::Vector{Float64}
	w::Vector{Float64}
end

(approx::AAAApprox)(z) = bary_eval(z, approx.f, approx.x, approx.w)

"""
	fit_aaa(x, y; tol=1e-13, mmax=100)

Fit AAA rational approximation.
"""
function fit_aaa(x, y; tol = 1e-13, mmax = 100)
	result = BaryRational.aaa(x, y; tol = tol, mmax = mmax, verbose = false)
	return AAAApprox(result.f, result.x, result.w)
end


# ============================================================================
# Analytic SE (RBF) GP with derivative evaluation (no AD)
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

# Gaussian Process Regression
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
	fit_gp_ad(x, y)

Fit GP using TaylorDiff-based automatic differentiation (simpler, more robust).
Based on ODEParameterEstimation approach - normalize, optimize, then AD through predictions.
"""
function fit_gp_ad(x::Vector{Float64}, y::Vector{Float64})
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
	kernel = SEIso(initial_lengthscale, initial_variance)
	jitter = 1e-8
	y_jittered = y_normalized .+ jitter * randn(length(y_normalized))

	# 2. Do GPR approximation on normalized data with suppressed warnings
	local gp
	@suppress gp = GP(x, y_jittered, MeanZero(), kernel, initial_noise)
	@suppress GaussianProcesses.optimize!(gp; method = LBFGS(linesearch = LineSearches.BackTracking()))

	# Create a function that evaluates the GPR prediction and denormalizes the output
	function denormalized_gpr(x_eval)
		pred, _ = predict_f(gp, [x_eval])
		return y_std * pred[1] + y_mean
	end

	return denormalized_gpr
end

"""
	fit_gp_matern(x, y; nu=1.5)

Fit GP with Matérn kernel using TaylorDiff-based automatic differentiation.
Uses same approach as GP-Julia-AD but with Matérn kernel instead of SE.

Supported nu values: 0.5 (Mat12Iso), 1.5 (Mat32Iso), 2.5 (Mat52Iso)
"""
function fit_gp_matern(x::Vector{Float64}, y::Vector{Float64}; nu::Float64 = 1.5)
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
	jitter = 1e-8
	y_jittered = y_normalized .+ jitter * randn(length(y_normalized))

	# 2. Do GPR approximation on normalized data with suppressed warnings
	local gp
	@suppress gp = GP(x, y_jittered, MeanZero(), kernel, initial_noise)
	@suppress GaussianProcesses.optimize!(gp; method = LBFGS(linesearch = LineSearches.BackTracking()))

	# Create a function that evaluates the GPR prediction and denormalizes the output
	function denormalized_gpr(x_eval)
		pred, _ = predict_f(gp, [x_eval])
		return y_std * pred[1] + y_mean
	end

	return denormalized_gpr
end


# ============================================================================
# Fourier Interpolation (from bary_derivs.jl)
# ============================================================================

"""Fourier series approximation with analytic derivatives"""
struct FourierSeries
	m::Float64
	b::Float64
	K::Float64
	cosines::Vector{Float64}
	sines::Vector{Float64}
end

"""Evaluate Fourier series at x"""
function (fs::FourierSeries)(x)
	z = fs.m * x + fs.b
	result = fs.K
	for (k, coef) in enumerate(fs.cosines)
		result += coef * cos(k * z)
	end
	for (k, coef) in enumerate(fs.sines)
		result += coef * sin(k * z)
	end
	return result
end

"""Evaluate n-th derivative of Fourier series at x (analytic)"""
function fourier_deriv(fs::FourierSeries, x::Float64, n::Int)
	if n == 0
		return fs(x)
	end

	z = fs.m * x + fs.b
	result = 0.0
	m_power = fs.m ^ n

	# Pattern: n=1: -k*sin, +k*cos
	#          n=2: -k²*cos, -k²*sin
	#          n=3: +k³*sin, -k³*cos
	#          n=4: +k⁴*cos, +k⁴*sin
	# Alternates every 2 derivatives

	for (k, a_k) in enumerate(fs.cosines)
		k_power = k ^ n
		if n % 4 == 1
			result += -k_power * a_k * sin(k * z)
		elseif n % 4 == 2
			result += -k_power * a_k * cos(k * z)
		elseif n % 4 == 3
			result += k_power * a_k * sin(k * z)
		else  # n % 4 == 0
			result += k_power * a_k * cos(k * z)
		end
	end

	for (k, b_k) in enumerate(fs.sines)
		k_power = k ^ n
		if n % 4 == 1
			result += k_power * b_k * cos(k * z)
		elseif n % 4 == 2
			result += -k_power * b_k * sin(k * z)
		elseif n % 4 == 3
			result += -k_power * b_k * cos(k * z)
		else  # n % 4 == 0
			result += k_power * b_k * sin(k * z)
		end
	end

	return m_power * result
end

"""
	fit_fourier(x, y; ridge_lambda=1e-8)

Fit Fourier interpolant using FFT-based spectral method.
For non-periodic data, uses symmetric extension to reduce edge effects.
"""
function fit_fourier(x, y; ridge_lambda = 1e-8)
	N = length(x)

	# Check if data is approximately uniform
	dx = diff(x)
	is_uniform = maximum(abs.(dx .- mean(dx))) / mean(dx) < 0.01

	if !is_uniform
		@warn "Fourier-Interp: Non-uniform spacing detected. Results may be less accurate."
	end

	# Symmetric extension to mitigate edge effects for non-periodic data
	# Extend by reflecting at boundaries
	y_extended = vcat(reverse(y[2:end]), y, reverse(y[1:end-1]))
	N_ext = length(y_extended)

	# Compute FFT
	y_fft = FFTW.fft(y_extended)

	# Wavenumber array for spectral differentiation
	# For uniform grid with spacing dx, sampling frequency fs = 1/dx
	# FFTW.fftfreq returns frequencies in cycles/unit
	# Multiply by 2π to get angular wavenumbers k (radians/unit)
	dx_mean = mean(diff(x))
	freqs = FFTW.fftfreq(N_ext, 1/dx_mean)  # Frequencies in cycles/unit
	k = 2π .* freqs  # Angular wavenumbers

	# Create interpolation data structure
	# Store FFT coefficients and grid info for later derivative computation
	x_min, x_max = x[1], x[end]

	# Return a custom struct with FFT data
	return FourierFFT(x_min, x_max, dx_mean, N, N_ext, y_fft, k, x, y)
end

"""Fourier approximation using FFT"""
struct FourierFFT
	x_min::Float64
	x_max::Float64
	dx::Float64
	N::Int
	N_ext::Int
	y_fft::Vector{ComplexF64}
	k::Vector{Float64}  # Angular wavenumbers (not frequencies)
	x_orig::Vector{Float64}
	y_orig::Vector{Float64}
end

"""Evaluate Fourier FFT at x"""
function (ff::FourierFFT)(x::Float64)
	# For evaluation, use simple interpolation of original data
	# This is more stable than IFFT for single points
	idx = searchsortedfirst(ff.x_orig, x)
	if idx == 1
		return ff.y_orig[1]
	elseif idx > ff.N
		return ff.y_orig[end]
	else
		# Linear interpolation
		t = (x - ff.x_orig[idx-1]) / (ff.x_orig[idx] - ff.x_orig[idx-1])
		return (1 - t) * ff.y_orig[idx-1] + t * ff.y_orig[idx]
	end
end

"""Evaluate n-th derivative using spectral differentiation with regularization"""
function fourier_fft_deriv(ff::FourierFFT, x::Float64, n::Int; filter_frac::Float64=0.4)
	if n == 0
		return ff(x)
	end

	# Compute derivative in Fourier space: multiply by (ik)^n
	# k already contains angular wavenumbers (2π × freq)
	deriv_fft = copy(ff.y_fft)

	# Regularization via low-pass filtering to suppress noise amplification
	# The (ik)^n operator acts as a high-pass filter, amplifying noise
	# We zero out high-frequency components to prevent catastrophic error growth
	k_max_abs = maximum(abs.(ff.k))
	k_cutoff = filter_frac * k_max_abs

	for i in eachindex(deriv_fft)
		k_abs = abs(ff.k[i])

		# Differentiate and apply filter simultaneously
		if k_abs <= k_cutoff
			deriv_fft[i] *= (im * ff.k[i])^n
		else
			# Zero out components above the cutoff frequency
			deriv_fft[i] = 0.0
		end
	end

	# Inverse FFT to get derivative in physical space
	deriv_extended = real(FFTW.ifft(deriv_fft))

	# Extract middle section (original domain, removing extension)
	offset = ff.N - 1
	deriv_orig = deriv_extended[(offset+1):(offset+ff.N)]

	# Interpolate at query point
	idx = searchsortedfirst(ff.x_orig, x)
	if idx == 1
		return deriv_orig[1]
	elseif idx > ff.N
		return deriv_orig[end]
	else
		# Linear interpolation
		t = (x - ff.x_orig[idx-1]) / (ff.x_orig[idx] - ff.x_orig[idx-1])
		return (1 - t) * deriv_orig[idx-1] + t * deriv_orig[idx]
	end
end


# ============================================================================
# Dierckx Spline
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
# Savitzky-Golay Filter
# ============================================================================

"""
	fit_savitzky_golay(x, y; window=11, polyorder=5, deriv=0)

Savitzky-Golay filter for smoothing and differentiation.
"""
function fit_savitzky_golay(x, y; window = 11, polyorder = 5)
	# Ensure window is odd
	window = isodd(window) ? window : window + 1
	window = min(window, length(y))

	# For derivatives, we'll use DSP.jl's built-in if available,
	# or implement a simple version
	smoothed = savitzky_golay_smooth(y, window, polyorder)

	# Return interpolator
	return z -> begin
		idx = argmin(abs.(x .- z))
		return smoothed[idx]
	end
end

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


# ============================================================================
# Lasso Trend Filtering
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
# Main Method Evaluation Function
# ============================================================================

"""
	evaluate_julia_method(method_name, x, y, x_eval, orders; params=Dict())

Evaluate a single Julia method on data.

Returns a MethodResult structure.
"""
function evaluate_julia_method(method_name::String,
	x::Vector{Float64},
	y::Vector{Float64},
	x_eval::Vector{Float64},
	orders::Vector{Int};
	params = Dict())

	t_start = time()
	predictions = Dict{Int, Vector{Float64}}()
	failures = Dict{Int, String}()

	try
		# Fit method
		fitted_func = nothing

		if method_name == "AAA-HighPrec"
			fitted_func = fit_aaa(x, y; tol = 1e-14)

		elseif method_name == "AAA-LowPrec"
			fitted_func = fit_aaa(x, y; tol = 0.1)

		elseif method_name == "GP-Julia-SE"
			# Use analytic SE GP for derivatives (stable up to high orders)
			gp_analytic = fit_gp_se_analytic(x, y)
			fitted_func = z -> gp_analytic(z, 0)

		elseif method_name == "GP-Julia-AD"
			# Use AD-based GP (simpler, more robust than analytic)
			fitted_func = fit_gp_ad(x, y)

		elseif method_name == "GP-Julia-Matern-0.5"
			# GP with Matérn-1/2 kernel (exponential covariance)
			fitted_func = fit_gp_matern(x, y; nu = 0.5)

		elseif method_name == "GP-Julia-Matern-1.5"
			# GP with Matérn-3/2 kernel (once differentiable)
			fitted_func = fit_gp_matern(x, y; nu = 1.5)

		elseif method_name == "GP-Julia-Matern-2.5"
			# GP with Matérn-5/2 kernel (twice differentiable)
			fitted_func = fit_gp_matern(x, y; nu = 2.5)

		elseif method_name == "Fourier-Interp"
			fitted_func = fit_fourier(x, y)

		elseif method_name == "Dierckx-5"
			fitted_func, _ = fit_dierckx_spline(x, y; k = 5,
				noise_level = get(params, :noise_level, 0.0))

		elseif method_name == "Savitzky-Golay"
			window = get(params, :sg_window, min(21, length(x)))
			polyorder = get(params, :sg_polyorder, 5)
			fitted_func = fit_savitzky_golay(x, y; window = window, polyorder = polyorder)

		elseif method_name == "TrendFilter-k7"
			λ = get(params, :tf_lambda, 0.1)
			fitted_func = fit_trend_filter(x, y; order = 7, λ = λ)

		elseif method_name == "TrendFilter-k2"
			λ = get(params, :tf_lambda, 0.1)
			fitted_func = fit_trend_filter(x, y; order = 2, λ = λ)

		elseif method_name == "TVRegDiff-Julia"
			# Total variation regularized differentiation (Rick Chartrand)
			# NOTE: Iterative differentiation is numerically unstable for orders > 1
			# Limit to first derivative only
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
			fitted_func = z -> begin
				# Linear interpolation of order-0 data
				idx = searchsortedfirst(x, z)
				if idx == 1
					return y[1]
				elseif idx > length(x)
					return y[end]
				else
					t = (z - x[idx-1]) / (x[idx] - x[idx-1])
					return (1 - t) * y[idx-1] + t * y[idx]
				end
			end

		elseif method_name == "Central-FD"
			fitted_func = fit_finite_diff(x, y)

		else
			error("Unknown method: $method_name")
		end

		# Compute derivatives
		if method_name == "Central-FD"
			# Special handling for FD
			for order in orders
				try
					predictions[order] = [fitted_func(xi, order) for xi in x_eval]
				catch e
					predictions[order] = fill(NaN, length(x_eval))
					failures[order] = string(e)
				end
			end
		elseif method_name == "Dierckx-5"
			# Dierckx has native derivative support
			_, spl = fit_dierckx_spline(x, y; k = 5,
				noise_level = get(params, :noise_level, 0.0))
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
		elseif method_name == "GP-Julia-SE"
			# Evaluate derivatives directly from analytic SE GP
			for order in orders
				try
					predictions[order] = [gp_analytic(xi, order) for xi in x_eval]
				catch e
					predictions[order] = fill(NaN, length(x_eval))
					failures[order] = string(e)
				end
			end
		elseif method_name == "Fourier-Interp"
			# Use FFT-based spectral differentiation with low-pass filtering for noise
			fourier_fft = fitted_func  # This is a FourierFFT struct
			filter_frac = get(params, :fourier_filter_frac, 0.4)
			for order in orders
				try
					predictions[order] = [fourier_fft_deriv(fourier_fft, xi, order; filter_frac=filter_frac) for xi in x_eval]
				catch e
					predictions[order] = fill(NaN, length(x_eval))
					failures[order] = string(e)
				end
			end
		elseif method_name == "TVRegDiff-Julia"
			# Use precomputed TV derivatives with linear interpolation
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
		else
			# Use TaylorDiff for automatic differentiation (all orders)
			for order in orders
				try
					predictions[order] = [nth_deriv_taylor(fitted_func, order, xi) for xi in x_eval]
				catch e
					predictions[order] = fill(NaN, length(x_eval))
					failures[order] = string(e)
				end
			end
		end

		timing = time() - t_start
		success = !isempty(predictions)

		# Determine category
		category = if method_name in ["AAA-HighPrec", "AAA-LowPrec"]
			"Rational"
		elseif method_name in ["GP-Julia-SE", "GP-Julia-AD", "GP-Julia-Matern-0.5", "GP-Julia-Matern-1.5", "GP-Julia-Matern-2.5"]
			"Gaussian Process"
		elseif method_name in ["Fourier-Interp"]
			"Spectral"
		elseif method_name in ["Dierckx-5"]
			"Spline"
		elseif method_name in ["Savitzky-Golay"]
			"Local Polynomial"
		elseif method_name in ["TrendFilter-k7", "TrendFilter-k2", "TVRegDiff-Julia"]
			"Regularization"
		elseif method_name in ["Central-FD"]
			"Finite Difference"
		else
			"Other"
		end

		return MethodResult(method_name, category, predictions, failures, timing, success)

	catch e
		@warn "Method $method_name failed" exception=e
		timing = time() - t_start
		return MethodResult(method_name, "Error", Dict(), Dict(0 => string(e)), timing, false)
	end
end


# ============================================================================
# Batch Evaluation
# ============================================================================

"""
	evaluate_all_julia_methods(x, y, x_eval, orders; params=Dict())

Evaluate all implemented Julia methods.
"""
function evaluate_all_julia_methods(x, y, x_eval, orders; params = Dict())
	methods = [
		"AAA-HighPrec",
		"AAA-LowPrec",
		"GP-Julia-SE",  # Re-enabled - analytic implementation is robust
		"GP-Julia-AD",  # AD-based GP (simpler, more robust)
		"GP-Julia-Matern-0.5",  # GP with Matérn-1/2 kernel
		"GP-Julia-Matern-1.5",  # GP with Matérn-3/2 kernel
		"GP-Julia-Matern-2.5",  # GP with Matérn-5/2 kernel
		"Fourier-Interp",
		"Dierckx-5",
		"Savitzky-Golay",
		"TrendFilter-k7",
		"TrendFilter-k2",
		"TVRegDiff-Julia",
		"Central-FD",
	]
	methods = [m for m in methods if m !== nothing]

	results = MethodResult[]

	for method in methods
		println("  Evaluating $method...")
		result = evaluate_julia_method(method, x, y, x_eval, orders; params = params)
		push!(results, result)
	end

	return results
end

