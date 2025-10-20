"""
Minimal Pilot - Single Configuration Test
"""

include("ground_truth.jl")
include("noise_model.jl")
include("julia_methods.jl")

using Random
using JSON3
using CSV
using DataFrames
using Printf

const PYTHON_SCRIPT = joinpath(@__DIR__, "..", "python", "python_methods.py")
const PYTHON_VENV = joinpath(@__DIR__, "..", "python", ".venv", "bin", "python")

println("=" ^ 70)
println("MINIMAL PILOT: Single Configuration Test")
println("=" ^ 70)

# Single configuration (env-overridable)
trial_id = "minimal_test"
noise_level = try
	parse(Float64, get(ENV, "NOISE", string(0.0)))
catch
	0.0
end
dsize = try
	parse(Int, get(ENV, "DSIZE", string(51)))
catch
	51
end
max_deriv = try
	parse(Int, get(ENV, "MAX_DERIV", string(3)))
catch
	3
end  # Lower for speed by default

# Generate ground truth
println("\nGenerating ground truth...")
sys = lotka_volterra_system()
times = collect(range(sys.tspan[1], sys.tspan[2], length = dsize))
truth = generate_ground_truth(sys, times, max_deriv)

# Noise control (use ConstantGaussian model by default)
y_true = truth[:obs][1][0]
if noise_level > 0
	y_noisy = add_noise(y_true, noise_level, Random.MersenneTwister(1234); model = ConstantGaussian)
else
	y_noisy = y_true
end

# Export to JSON for Python
input_json_path = joinpath(@__DIR__, "..", "data", "input", "$(trial_id).json")
input_data = Dict(
	"system" => "Lotka-Volterra",
	"observable" => "x(t)",
	"times" => times,
	"y_noisy" => y_noisy,
	"y_true" => y_true,
	"ground_truth_derivatives" => Dict(
		string(order) => truth[:obs][1][order]
		for order in 0:max_deriv
	),
	"config" => Dict(
		"trial_id" => trial_id,
		"data_size" => dsize,
		"noise_level" => noise_level,
		"trial" => 1,
		"tspan" => [sys.tspan[1], sys.tspan[2]],
	),
)

open(input_json_path, "w") do f
	JSON3.write(f, input_data)
end

# Call Python script with timeout
output_json_path = joinpath(@__DIR__, "..", "data", "output", "$(trial_id)_results.json")
cmd = `$PYTHON_VENV $PYTHON_SCRIPT $input_json_path $output_json_path`

println("Running Python methods...")
# Timeout in seconds (configurable via environment variable)
timeout_sec = try
	parse(Int, get(ENV, "PYTHON_TIMEOUT", string(300)))
catch
	300  # Default: 5 minutes
end

python_task = @async try
	run(cmd)
	:success
catch e
	(:error, e)
end

# Wait for completion or timeout
result = timedwait(() -> istaskdone(python_task), timeout_sec; pollint=0.5)

if result == :timed_out
	@warn "Python script timed out after $(timeout_sec) seconds, terminating..."
	# Note: Julia doesn't provide direct process killing, so this is best-effort
	# The subprocess may continue running in background
else
	task_result = fetch(python_task)
	if task_result == :success
		println("Python methods completed successfully")
	else
		@warn "Python script failed" exception=task_result[2]
	end
end

# Evaluate Julia methods
println("Running Julia methods...")
orders = collect(0:max_deriv)
julia_results = evaluate_all_julia_methods(
	times, y_noisy, times, orders;
	params = Dict(:noise_level => noise_level),
)

# Collect results
all_results = []

# Process Julia results
for result in julia_results
	if result.success
		for order in orders
			if haskey(result.predictions, order)
				pred = result.predictions[order]
				true_vals = truth[:obs][1][order]

				valid = .!isnan.(pred) .& .!isinf.(pred)
				if sum(valid) > 2
					# Exclude endpoints
					idxrng = 2:(length(pred)-1)
					vmask = valid[idxrng]
					if sum(vmask) > 0
						rmse = sqrt(mean((pred[idxrng][vmask] .- true_vals[idxrng][vmask]) .^ 2))
						mae = mean(abs.(pred[idxrng][vmask] .- true_vals[idxrng][vmask]))
					else
						rmse = NaN;
						mae = NaN
					end

					push!(all_results, (
						method = result.name,
						category = result.category,
						language = "Julia",
						deriv_order = order,
						rmse = rmse,
						mae = mae,
						timing = result.timing,
						valid_points = sum(valid),
					))
				end
			end
		end
	end
end

# Process Python results
if isfile(output_json_path)
	python_output = JSON3.read(read(output_json_path, String))

	for (method_name, method_result) in python_output["methods"]
		if method_result["success"]
			for order in orders
				if haskey(method_result["predictions"], string(order))
					pred = method_result["predictions"][string(order)]
					true_vals = truth[:obs][1][order]

					valid = .!isnan.(pred) .& .!isinf.(pred)
					if sum(valid) > 2
						# Exclude endpoints
						idxrng = 2:(length(pred)-1)
						vmask = valid[idxrng]
						if sum(vmask) > 0
							rmse = sqrt(mean((pred[idxrng][vmask] .- true_vals[idxrng][vmask]) .^ 2))
							mae = mean(abs.(pred[idxrng][vmask] .- true_vals[idxrng][vmask]))
						else
							rmse = NaN;
							mae = NaN
						end

						method_str = string(method_name)
						category = if contains(method_str, "GP")
							"Gaussian Process"
						elseif contains(method_str, "RBF")
							"RBF"
						elseif contains(method_str, "Spline")
							"Spline"
						elseif contains(method_str, "FD")
							"Finite Difference"
						else
							"Other"
						end

						push!(all_results, (
							method = method_str,
							category = category,
							language = "Python",
							deriv_order = order,
							rmse = rmse,
							mae = mae,
							timing = method_result["timing"],
							valid_points = sum(valid),
						))
					end
				end
			end
		end
	end
end

# Save results
println("\nSaving results...")
df = DataFrame(all_results)
results_dir = joinpath(@__DIR__, "..", "results", "pilot")
mkpath(results_dir)
CSV.write(joinpath(results_dir, "minimal_pilot_results.csv"), df)

# Print summary
println("\n" * "=" ^ 70)
println("MINIMAL PILOT COMPLETE")
println("=" ^ 70)
println("\nResults for order 3:")
order3 = df[df.deriv_order .== 3, :]
sort!(order3, :rmse)
for i in 1:nrow(order3)
	row = order3[i, :]
	println(@sprintf("  %-30s RMSE=%.6f  MAE=%.6f  (%.3fs)",
		row.method, row.rmse, row.mae, row.timing))
end

println("\nResults saved to: $(results_dir)/minimal_pilot_results.csv")
println("=" ^ 70)
