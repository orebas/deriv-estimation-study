"""
Julia-based Differentiation Methods (Integrated from Extracted Modules)

This file integrates all extracted Julia methods from methods/julia/* into the
main benchmark pipeline, replacing the monolithic julia_methods.jl.
"""

# Load hyperparameter selection module
include("hyperparameter_selection.jl")
using .HyperparameterSelection

# Load all extracted method modules
include("../methods/julia/common.jl")
include("../methods/julia/gp/gaussian_process.jl")
include("../methods/julia/rational/aaa.jl")
include("../methods/julia/spectral/fourier.jl")
include("../methods/julia/splines/splines.jl")
include("../methods/julia/filtering/filters.jl")
include("../methods/julia/regularization/regularized.jl")
include("../methods/julia/finite_diff/finite_diff.jl")

# ============================================================================
# Batch Evaluation (Main API for Pipeline)
# ============================================================================

"""
	evaluate_all_julia_methods(x, y, x_eval, orders; params=Dict())

Evaluate all implemented Julia methods using extracted modules.

This function maintains the same API as the original monolithic implementation
but now calls the extracted, organized methods.
"""
function evaluate_all_julia_methods(x, y, x_eval, orders; params = Dict())
	# Map of method names to evaluation functions
	method_map = Dict(
		# GP methods (5 methods)
		"GP-Julia-SE" => evaluate_gp_se,
		"GP-Julia-AD" => evaluate_gp_ad,
		"GP-Julia-Matern-0.5" => evaluate_gp_matern_05,
		"GP-Julia-Matern-1.5" => evaluate_gp_matern_15,
		"GP-Julia-Matern-2.5" => evaluate_gp_matern_25,

		# AAA/Rational methods (4 methods)
		"AAA-HighPrec" => evaluate_aaa_highprec,
		"AAA-LowPrec" => evaluate_aaa_lowprec,
		"AAA-Adaptive-Diff2" => evaluate_aaa_adaptive_diff2,
		"AAA-Adaptive-Wavelet" => evaluate_aaa_adaptive_wavelet,

		# Spectral methods (2 methods)
		"Fourier-Interp" => evaluate_fourier_interp,
		"Fourier-FFT-Adaptive" => evaluate_fourier_fft_adaptive,

		# Splines (2 methods)
		"Dierckx-5" => evaluate_dierckx,
		"GSS" => evaluate_gss,

		# Filtering (1 method)
		"Savitzky-Golay" => evaluate_savitzky_golay,

		# Regularization (3 methods)
		"TrendFilter-k7" => evaluate_trend_filter_k7,
		"TrendFilter-k2" => evaluate_trend_filter_k2,
		"TVRegDiff-Julia" => evaluate_tvregdiff,

		# Finite Diff (1 method)
		"Central-FD" => evaluate_central_fd,
	)

	# Define which methods to run (matching original behavior)
	methods_to_run = [
		"AAA-HighPrec",
		"AAA-LowPrec",
		"AAA-Adaptive-Diff2",
		"AAA-Adaptive-Wavelet",
		# "GP-Julia-SE" removed per user request
		"GP-Julia-AD",  # AD-based GP (simpler, more robust)
		# Matérn methods disabled by default (can be enabled)
		# "GP-Julia-Matern-0.5",
		# "GP-Julia-Matern-1.5",
		# "GP-Julia-Matern-2.5",
		"Fourier-Interp",
		"Fourier-FFT-Adaptive",
		"Dierckx-5",
		"GSS",
		"Savitzky-Golay",
		"TrendFilter-k7",
		"TrendFilter-k2",
		"TVRegDiff-Julia",
		"Central-FD",
	]

	results = MethodResult[]

	for method_name in methods_to_run
		println("  Evaluating $method_name...")

		if !haskey(method_map, method_name)
			@warn "Method $method_name not found in method_map, skipping"
			continue
		end

		try
			eval_func = method_map[method_name]
			result = eval_func(x, y, x_eval, orders; params = params)
			push!(results, result)
		catch e
			@warn "Method $method_name failed during evaluation" exception=(e, catch_backtrace())
			# Create error result
			error_result = MethodResult(
				method_name,
				"Error",
				Dict{Int, Vector{Float64}}(),
				Dict(0 => string(e)),
				0.0,
				false
			)
			push!(results, error_result)
		end
	end

	return results
end

# For backward compatibility: provide evaluate_julia_method for single method evaluation
"""
	evaluate_julia_method(method_name, x, y, x_eval, orders; params=Dict())

Evaluate a single Julia method by name (backward compatibility wrapper).
"""
function evaluate_julia_method(method_name::String, x, y, x_eval, orders; params = Dict())
	# Call evaluate_all with just this one method, then return the first result
	results = evaluate_all_julia_methods(x, y, x_eval, orders; params = params)

	# Find the result for this method
	for result in results
		if result.name == method_name
			return result
		end
	end

	# If not found, return error
	return MethodResult(
		method_name,
		"Error",
		Dict{Int, Vector{Float64}}(),
		Dict(0 => "Method not found: $method_name"),
		0.0,
		false
	)
end
