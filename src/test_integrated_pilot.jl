"""
Test Integrated Pilot - Tests the integrated extracted methods

This script tests the integrated extracted methods to ensure they work
correctly in the pipeline before we replace the original files.
"""

include("ground_truth.jl")
include("noise_model.jl")
include("julia_methods_integrated.jl")  # Use integrated methods instead of julia_methods.jl

using Random
using JSON3
using CSV
using DataFrames
using Printf

const PYTHON_SCRIPT = joinpath(@__DIR__, "..", "python", "python_methods_integrated.py")  # Use integrated Python script
const PYTHON_VENV = joinpath(@__DIR__, "..", "python", ".venv", "bin", "python")

println("=" ^ 70)
println("TEST INTEGRATED PILOT: Testing Extracted Methods")
println("=" ^ 70)

# Single configuration (env-overridable)
trial_id = "test_integrated"
noise_level = try
	parse(Float64, get(ENV, "NOISE", string(0.01)))
catch
	0.01
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
end

# Generate ground truth
println("\nGenerating ground truth...")
sys = lotka_volterra_system()
times = collect(range(sys.tspan[1], sys.tspan[2], length = dsize))
truth = generate_ground_truth(sys, times, max_deriv)

# Noise control
y_true = truth[:obs][1][0]
if noise_level > 0
	y_noisy = add_noise(y_true, noise_level, Random.MersenneTwister(1234); model = ConstantGaussian)
else
	y_noisy = y_true
end

# Export to JSON for Python
input_json_path = joinpath(@__DIR__, "..", "build", "data", "input", "$(trial_id).json")
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
output_json_path = joinpath(@__DIR__, "..", "build", "data", "output", "$(trial_id)_results.json")
cmd = `$PYTHON_VENV $PYTHON_SCRIPT $input_json_path $output_json_path`

println("Running Python methods (INTEGRATED)...")
timeout_sec = try
	parse(Int, get(ENV, "PYTHON_TIMEOUT", string(300)))
catch
	300
end

python_task = @async try
	run(cmd)
	:success
catch e
	(:error, e)
end

result = timedwait(() -> istaskdone(python_task), timeout_sec; pollint=0.5)

if result == :timed_out
	@warn "Python script timed out after $(timeout_sec) seconds"
else
	task_result = fetch(python_task)
	if task_result == :success
		println("Python methods completed successfully")
	else
		@warn "Python script failed" exception=task_result[2]
	end
end

# Evaluate Julia methods (INTEGRATED)
println("Running Julia methods (INTEGRATED)...")
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
						valid_points = sum(vmask),
					))
				end
			end
		end
	end
end

# Process Python results if available
if isfile(output_json_path)
	py_data = JSON3.read(read(output_json_path, String))
	if haskey(py_data, :methods)
		for (method_name, method_result) in py_data[:methods]
			if haskey(method_result, :predictions)
				for (order_str, pred) in method_result[:predictions]
					order = parse(Int, String(order_str))
					if order in orders
						true_vals = truth[:obs][1][order]
						pred_arr = Float64.(pred)

						valid = .!isnan.(pred_arr) .& .!isinf.(pred_arr)
						if sum(valid) > 2
							idxrng = 2:(length(pred_arr)-1)
							vmask = valid[idxrng]
							if sum(vmask) > 0
								rmse = sqrt(mean((pred_arr[idxrng][vmask] .- true_vals[idxrng][vmask]) .^ 2))
								mae = mean(abs.(pred_arr[idxrng][vmask] .- true_vals[idxrng][vmask]))

								push!(all_results, (
									method = string(method_name),
									category = "Python",
									language = "Python",
									deriv_order = order,
									rmse = rmse,
									mae = mae,
									timing = get(method_result, :timing, NaN),
									valid_points = sum(vmask),
								))
							end
						end
					end
				end
			end
		end
	end
end

# Save results
output_dir = joinpath(@__DIR__, "..", "build", "results", "test_integrated")
mkpath(output_dir)
output_csv = joinpath(output_dir, "test_integrated_results.csv")

df = DataFrame(all_results)
CSV.write(output_csv, df)

println("\n" * "=" ^ 70)
println("RESULTS SUMMARY")
println("=" ^ 70)
println("Total methods evaluated: $(length(unique(df.method)))")
println("Total results: $(nrow(df))")
println("Results saved to: $output_csv")
println("=" ^ 70)

# Show sample results
println("\nTop 10 results by RMSE (order 1):")
order1_results = filter(row -> row.deriv_order == 1, df)
if nrow(order1_results) > 0
	sort!(order1_results, :rmse)
	for row in eachrow(first(order1_results, min(10, nrow(order1_results))))
		@printf("  %-40s RMSE: %.6f  Lang: %s\n", row.method, row.rmse, row.language)
	end
else
	println("  No order-1 results found")
end

println("\n✅ Test integrated pilot completed successfully!")
