"""
Spectral Fourier Methods for Derivative Estimation

Implements 2 FFT-based methods:
1. Fourier-Interp: FFT-based spectral differentiation with fixed low-pass filtering
2. Fourier-FFT-Adaptive: FFT-based spectral differentiation with adaptive noise-based filtering

Both use symmetric extension to mitigate edge effects for non-periodic data.
"""

# Load common utilities
include("../common.jl")

using FFTW

# ============================================================================
# Fourier FFT Data Structure
# ============================================================================

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
# Method Evaluators (Standard API)
# ============================================================================

"""
	evaluate_fourier_interp(x, y, x_eval, orders; params=Dict())

Evaluate Fourier-Interp method: FFT-based spectral differentiation with fixed low-pass filtering.
"""
function evaluate_fourier_interp(
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
		# Fit Fourier FFT
		fitted_func = fit_fourier(x, y)

		# Get filter fraction from params or environment
		filter_frac = get(params, :fourier_filter_frac, 0.4)

		# Use FFT-based spectral differentiation with fixed low-pass filtering
		for order in orders
			try
				predictions[order] = [fourier_fft_deriv(fitted_func, xi, order; filter_frac=filter_frac) for xi in x_eval]
			catch e
				predictions[order] = fill(NaN, length(x_eval))
				failures[order] = string(e)
			end
		end

		timing = time() - t_start
		return MethodResult("Fourier-Interp", "Spectral", predictions, failures, timing, true)

	catch e
		@warn "Fourier-Interp failed" exception=e
		timing = time() - t_start
		return MethodResult("Fourier-Interp", "Error", Dict(), Dict(0 => string(e)), timing, false)
	end
end

"""
	evaluate_fourier_fft_adaptive(x, y, x_eval, orders; params=Dict())

Evaluate Fourier-FFT-Adaptive method: FFT-based spectral differentiation with adaptive noise-based filtering.
"""
function evaluate_fourier_fft_adaptive(
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
		# Fit Fourier FFT
		fitted_func = fit_fourier(x, y)

		# Adaptive filter fraction based on noise estimation
		filter_frac = HyperparameterSelection.select_fourier_filter_frac(y)

		# Use FFT-based spectral differentiation with adaptive filtering
		for order in orders
			try
				predictions[order] = [fourier_fft_deriv(fitted_func, xi, order; filter_frac=filter_frac) for xi in x_eval]
			catch e
				predictions[order] = fill(NaN, length(x_eval))
				failures[order] = string(e)
			end
		end

		timing = time() - t_start
		return MethodResult("Fourier-FFT-Adaptive", "Spectral", predictions, failures, timing, true)

	catch e
		@warn "Fourier-FFT-Adaptive failed" exception=e
		timing = time() - t_start
		return MethodResult("Fourier-FFT-Adaptive", "Error", Dict(), Dict(0 => string(e)), timing, false)
	end
end
