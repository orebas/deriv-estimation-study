#!/usr/bin/env julia

using JSON3
using Statistics
using Printf

# Include Julia GP methods (brings in common utilities via relative includes)
include("../methods/julia/gp/gaussian_process.jl")

function main()
	if length(ARGS) < 1
		println("Usage: julia repro_julia_gp.jl <input_json>")
		exit(1)
	end
	input_path = ARGS[1]
	data = JSON3.read(read(input_path, String))

	# Extract data
	times = Vector{Float64}(data["times"])
	y_noisy = Vector{Float64}(data["y_noisy"])
	y_true0 = Vector{Float64}(data["ground_truth_derivatives"]["0"])

	orders = collect(0:7)

	# Run GP-Julia-AD
	@printf "Running GP-Julia-AD on %s\n" input_path
	result = evaluate_gp_ad(times, y_noisy, times, orders; params = Dict())

	# Summarize
	failures = result.failures
	println("Failures:", failures)

	# Compute error for order 0 like the study (exclude endpoints, finite mask)
	if haskey(result.predictions, 0)
		pred0 = result.predictions[0]
		valid = .!isnan.(pred0) .& .!isinf.(pred0)
		if sum(valid) > 2
			idxrng = 2:(length(pred0)-1)
			vmask = valid[idxrng]
			rmse = sqrt(mean((pred0[idxrng][vmask] .- y_true0[idxrng][vmask]) .^ 2))
			mae = mean(abs.(pred0[idxrng][vmask] .- y_true0[idxrng][vmask]))
			true_std = std(y_true0[idxrng][vmask])
			nrmse = rmse / max(true_std, 1e-12)
			@printf "order0: rmse=%.6g mae=%.6g nrmse=%.6g\n" rmse mae nrmse
		else
			println("Insufficient valid points to compute error")
		end
	else
		println("No order 0 prediction")
	end

	println("Done.")
end

main()
