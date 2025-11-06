"""
Comprehensive Derivative Estimation Study

Runs extensive tests across multiple noise levels and derivative orders.
"""

include("ground_truth.jl")
include("noise_model.jl")
include("julia_methods_integrated.jl")
include("config_loader.jl")

using Random
using JSON3
using CSV
using DataFrames
using Printf
using Base.Threads

# Load study configuration from config.toml (single source of truth)
const CONFIG = get_comprehensive_config()
const NOISE_LEVELS = CONFIG.noise_levels
const DATA_SIZE = CONFIG.data_size
const MAX_DERIV = CONFIG.max_derivative_order
const TRIALS_PER_CONFIG = CONFIG.trials_per_config
const PYTHON_SCRIPT = joinpath(@__DIR__, "..", "python", "python_methods_integrated.py")
const PYTHON_VENV = joinpath(@__DIR__, "..", "python", ".venv", "bin", "python")

println("=" ^ 80)
println("COMPREHENSIVE DERIVATIVE ESTIMATION STUDY")
println("=" ^ 80)
print_config_summary(:comprehensive)

# Load enabled ODE systems from config
enabled_ode_keys = get_enabled_ode_systems()
println("\nEnabled ODE systems: $(join(enabled_ode_keys, ", "))")
ode_systems = get_all_ode_systems(enabled_ode_keys)
println("Loaded $(length(ode_systems)) ODE system(s)")

# Run comprehensive study
# Create flat configuration list for parallel execution
configs = [(ode_key, noise_level, trial)
		   for ode_key in enabled_ode_keys
		   for noise_level in NOISE_LEVELS
		   for trial in 1:TRIALS_PER_CONFIG]
total_configs = length(configs)

# Pre-allocate results array (use Any to accept NamedTuples)
all_results_array = Vector{Vector{Any}}(undef, total_configs)

# Thread-safe lock for logging and file I/O
io_lock = ReentrantLock()

println("\nRunning $total_configs configurations with $(Threads.nthreads()) threads in parallel...\n")

@threads for config_idx in 1:total_configs
	ode_key, noise_level, trial = configs[config_idx]
	config_results = []  # Thread-local results
	trial_id = "$(ode_key)_noise$(Int(noise_level*1e8))e-8_trial$(trial)"

	lock(io_lock) do
		println("[$(config_idx)/$total_configs] Thread $(threadid()): Processing ODE=$(ode_key), noise=$(noise_level), trial=$trial...")
	end

	# Generate ground truth for this ODE system
	sys_def = ode_systems[ode_key]
	times = collect(range(sys_def.tspan[1], sys_def.tspan[2], length = DATA_SIZE))
	truth = generate_ground_truth(sys_def, times, MAX_DERIV)

	# Add noise (use unique seed per config for thread safety)
	# Changed seed from 54321 to 98765 to generate different noise realizations
	rng = MersenneTwister(98765 + trial + config_idx * 1000)
	noisy = add_noise_to_data(truth, noise_level, rng; model = ConstantGaussian)

	# Extract observable 1
	y_true = truth[:obs][1][0]
	y_noisy = noisy[:obs][1][0]

	# === Export to JSON for Python ===
	input_json_path = joinpath(@__DIR__, "..", "build", "data", "input", "$(trial_id).json")
	input_data = Dict(
		"system" => sys_def.name,
		"ode_key" => ode_key,
		"observable" => "x(t)",
		"times" => times,
		"y_noisy" => y_noisy,
		"y_true" => y_true,
		"ground_truth_derivatives" => Dict(
			string(order) => truth[:obs][1][order]
			for order in 0:MAX_DERIV
		),
		"config" => Dict(
			"trial_id" => trial_id,
			"data_size" => DATA_SIZE,
			"noise_level" => noise_level,
			"trial" => trial,
			"tspan" => [sys_def.tspan[1], sys_def.tspan[2]],
		),
	)

	open(input_json_path, "w") do f
		JSON3.write(f, input_data)
	end

	# === Call Python script with timeout ===
	output_json_path = joinpath(@__DIR__, "..", "build", "data", "output", "$(trial_id)_results.json")
	cmd = `$PYTHON_VENV $PYTHON_SCRIPT $input_json_path $output_json_path`

	lock(io_lock) do
		println("  Running Python methods...")
	end
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

	result = timedwait(() -> istaskdone(python_task), timeout_sec; pollint = 0.5)

	if result == :timed_out
		@warn "Python script timed out for $trial_id"
	else
		task_result = fetch(python_task)
		if task_result != :success
			@warn "Python script failed for $trial_id" exception=task_result[2]
		end
	end

	# === Evaluate Julia methods ===
	lock(io_lock) do
		println("  Running Julia methods...")
	end
	orders = collect(0:MAX_DERIV)
	julia_results = evaluate_all_julia_methods(
		times, y_noisy, times, orders;
		params = Dict(),
	)

	# === Compute errors ===
	lock(io_lock) do
		println("  Computing errors...")
	end

	# Process Julia results
	for result in julia_results
		if result.success
			for order in orders
				if haskey(result.predictions, order)
					pred = result.predictions[order]
					true_vals = truth[:obs][1][order]

					# Compute RMSE excluding endpoints
					valid = .!isnan.(pred) .& .!isinf.(pred)
					if sum(valid) > 2
						# Exclude first and last points
						idxrng = 2:(length(pred)-1)
						vmask = valid[idxrng]
						if sum(vmask) > 0
							rmse = sqrt(mean((pred[idxrng][vmask] .- true_vals[idxrng][vmask]) .^ 2))
							mae = mean(abs.(pred[idxrng][vmask] .- true_vals[idxrng][vmask]))

							# Compute normalized RMSE (nRMSE = RMSE / std(true))
							true_std = std(true_vals[idxrng][vmask])
							nrmse = rmse / max(true_std, 1e-12)  # Avoid division by near-zero

							push!(
								config_results,
								(
									ode_system = ode_key,
									noise_level = noise_level,
									trial = trial,
									method = result.name,
									category = result.category,
									language = "Julia",
									deriv_order = order,
									rmse = rmse,
									mae = mae,
									nrmse = nrmse,
									timing = result.timing,
									valid_points = sum(vmask),
									total_points = length(idxrng),
								),
							)
						end
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

								# Compute normalized RMSE (nRMSE = RMSE / std(true))
								true_std = std(true_vals[idxrng][vmask])
								nrmse = rmse / max(true_std, 1e-12)  # Avoid division by near-zero

								# Determine category
								method_str = string(method_name)
								category = if contains(method_str, "GP")
									"Gaussian Process"
								elseif contains(method_str, "RBF")
									"RBF"
								elseif contains(method_str, "Spline")
									"Spline"
								elseif contains(method_str, "FD")
									"Finite Difference"
								elseif contains(method_str, "Fourier") || contains(method_str, "Chebyshev")
									"Spectral"
								else
									"Other"
								end

								push!(
									config_results,
									(
										ode_system = ode_key,
										noise_level = noise_level,
										trial = trial,
										method = method_str,
										category = category,
										language = "Python",
										deriv_order = order,
										rmse = rmse,
										mae = mae,
										nrmse = nrmse,
										timing = method_result["timing"],
										valid_points = sum(vmask),
										total_points = length(idxrng),
									),
								)
							end
						end
					end
				end
			end
		end
	end

	# === Save combined predictions JSON (for visualization) ===
	predictions_dir = joinpath(@__DIR__, "..", "build", "results", "comprehensive", "predictions")
	mkpath(predictions_dir)

	# Helper function to convert NaN/Inf to nothing for JSON compatibility
	clean_for_json(x::AbstractArray) = map(v -> isfinite(v) ? v : nothing, x)
	clean_for_json(x) = x

	# Combine Python and Julia predictions
	combined_methods = Dict{String, Any}()

	# Add Python methods
	if isfile(output_json_path)
		python_output = JSON3.read(read(output_json_path, String))
		for (method_name, method_result) in python_output["methods"]
			# Clean predictions (convert NaN/Inf to null)
			cleaned_preds = Dict(
				string(k) => clean_for_json(v)
				for (k, v) in method_result["predictions"]
			)
			combined_methods[string(method_name)] = Dict(
				"predictions" => cleaned_preds,
				"timing" => method_result["timing"],
				"success" => method_result["success"],
				"language" => "Python",
			)
		end
	end

	# Add Julia methods
	for result in julia_results
		# Clean predictions (convert NaN/Inf to null)
		cleaned_preds = Dict(
			string(k) => clean_for_json(v)
			for (k, v) in result.predictions
		)
		combined_methods[result.name] = Dict(
			"predictions" => cleaned_preds,
			"timing" => result.timing,
			"success" => result.success,
			"language" => "Julia",
		)
	end

	# Create combined JSON (also clean ground truth)
	predictions_json = Dict(
		"trial_id" => trial_id,
		"config" => Dict(
			"noise_level" => noise_level,
			"trial" => trial,
			"data_size" => DATA_SIZE,
		),
		"times" => times,
		"ground_truth_derivatives" => Dict(
			string(order) => clean_for_json(truth[:obs][1][order])
			for order in 0:MAX_DERIV
		),
		"methods" => combined_methods,
	)

	# Write to predictions directory
	predictions_path = joinpath(predictions_dir, "$(trial_id).json")
	open(predictions_path, "w") do f
		JSON3.write(f, predictions_json)
	end

	# Store thread-local results into shared array
	all_results_array[config_idx] = config_results
end  # End of @threads loop

# Flatten results from parallel execution
all_results = vcat(all_results_array...)

# Save results
println("\nSaving results...")
df = DataFrame(all_results)
results_dir = joinpath(@__DIR__, "..", "build", "results", "comprehensive")
mkpath(results_dir)
CSV.write(joinpath(results_dir, "comprehensive_results.csv"), df)

# Create summary statistics (grouped by ODE system)
println("Creating summary statistics...")
summary = combine(groupby(df, [:ode_system, :method, :category, :language, :deriv_order, :noise_level])) do sdf
	(
		mean_rmse = mean(sdf.rmse),
		std_rmse = std(sdf.rmse),
		min_rmse = minimum(sdf.rmse),
		max_rmse = maximum(sdf.rmse),
		mean_mae = mean(sdf.mae),
		mean_nrmse = mean(sdf.nrmse),
		std_nrmse = std(sdf.nrmse),
		min_nrmse = minimum(sdf.nrmse),
		max_nrmse = maximum(sdf.nrmse),
		mean_timing = mean(sdf.timing),
		trials = nrow(sdf),
	)
end

CSV.write(joinpath(results_dir, "comprehensive_summary.csv"), summary)

# Helper: find missing (ode, noise, trial) configs with no rows in df
function _find_missing_configs(df::DataFrame,
	ode_keys::Vector{String},
	noise_levels::Vector{Float64},
	trials_per_config::Int)
	present = Set{Tuple{String, Float64, Int}}()
	for row in eachrow(df)
		push!(present, (String(row.ode_system), Float64(row.noise_level), Int(row.trial)))
	end
	missing = Tuple{String, Float64, Int}[]
	for ode in ode_keys
		for nl in noise_levels
			for t in 1:trials_per_config
				key = (ode, nl, t)
				if !(key in present)
					push!(missing, key)
				end
			end
		end
	end
	return missing
end

# Helper: retry a single config by reusing existing input JSON (deterministic)
function _retry_config_once(ode_key::String,
	noise_level::Float64,
	trial::Int,
	orders::Vector{Int},
	py_exe::String,
	py_script::String)
	results_accum = Vector{NamedTuple}()
	trial_id = "$(ode_key)_noise$(Int(noise_level*1e8))e-8_trial$(trial)"
	input_json_path = joinpath(@__DIR__, "..", "build", "data", "input", "$(trial_id).json")
	output_json_path = joinpath(@__DIR__, "..", "build", "data", "output", "$(trial_id)_results.json")

	if !isfile(input_json_path)
		@warn "Retry skipped: input JSON not found" input=input_json_path
		return results_accum
	end

	# Read input JSON to get deterministic data
	input_data = JSON3.read(read(input_json_path, String))
	times = Vector{Float64}(input_data["times"])
	y_noisy = Vector{Float64}(input_data["y_noisy"])
	# Ground-truth derivatives from JSON for error calc
	gt_derivs = Dict(parse(Int, k) => Vector{Float64}(v) for (k, v) in input_data["ground_truth_derivatives"])

	# Re-run Python script
	try
		run(`$py_exe $py_script $input_json_path $output_json_path`)
	catch e
		@warn "Retry: Python script failed" exception=(e,)
	end

	# Evaluate Julia methods
	julia_results = evaluate_all_julia_methods(times, y_noisy, times, orders; params = Dict())

	# Compute errors for Julia results
	for result in julia_results
		if result.success
			for order in orders
				if haskey(result.predictions, order)
					pred = result.predictions[order]
					true_vals = gt_derivs[order]

					valid = .!isnan.(pred) .& .!isinf.(pred)
					if sum(valid) > 2
						idxrng = 2:(length(pred)-1)
						vmask = valid[idxrng]
						if sum(vmask) > 0
							rmse = sqrt(mean((pred[idxrng][vmask] .- true_vals[idxrng][vmask]) .^ 2))
							mae = mean(abs.(pred[idxrng][vmask] .- true_vals[idxrng][vmask]))
							true_std = std(true_vals[idxrng][vmask])
							nrmse = rmse / max(true_std, 1e-12)

							push!(
								results_accum,
								(
									ode_system = ode_key,
									noise_level = noise_level,
									trial = trial,
									method = result.name,
									category = result.category,
									language = "Julia",
									deriv_order = order,
									rmse = rmse,
									mae = mae,
									nrmse = nrmse,
									timing = result.timing,
									valid_points = sum(vmask),
									total_points = length(idxrng),
								),
							)
						end
					end
				end
			end
		end
	end

	# Process Python results (if produced)
	if isfile(output_json_path)
		py_out = JSON3.read(read(output_json_path, String))
		for (method_name, method_result) in py_out["methods"]
			if method_result["success"]
				for order in orders
					if haskey(method_result["predictions"], string(order))
						pred = method_result["predictions"][string(order)]
						true_vals = gt_derivs[order]

						valid = .!isnan.(pred) .& .!isinf.(pred)
						if sum(valid) > 2
							idxrng = 2:(length(pred)-1)
							vmask = valid[idxrng]
							if sum(vmask) > 0
								rmse = sqrt(mean((pred[idxrng][vmask] .- true_vals[idxrng][vmask]) .^ 2))
								mae = mean(abs.(pred[idxrng][vmask] .- true_vals[idxrng][vmask]))
								true_std = std(true_vals[idxrng][vmask])
								nrmse = rmse / max(true_std, 1e-12)

								method_str = string(method_name)
								category = if contains(method_str, "GP")
									"Gaussian Process"
								elseif contains(method_str, "RBF")
									"RBF"
								elseif contains(method_str, "Spline")
									"Spline"
								elseif contains(method_str, "FD")
									"Finite Difference"
								elseif contains(method_str, "Fourier") || contains(method_str, "Chebyshev")
									"Spectral"
								else
									"Other"
								end

								push!(
									results_accum,
									(
										ode_system = ode_key,
										noise_level = noise_level,
										trial = trial,
										method = method_str,
										category = category,
										language = "Python",
										deriv_order = order,
										rmse = rmse,
										mae = mae,
										nrmse = nrmse,
										timing = method_result["timing"],
										valid_points = sum(vmask),
										total_points = length(idxrng),
									),
								)
							end
						end
					end
				end
			end
		end
	end

	return results_accum
end

# Create comprehensive failure report (includes methods that completely failed)
println("Creating failure report...")

# Step 1: Collect all Julia methods that were attempted
julia_methods_attempted = [
	"AAA-LowPrec",
	"AAA-Adaptive-Diff2",
	"AAA-Adaptive-Wavelet",
	"GP-Julia-AD",
	"Fourier-Interp",
	"Fourier-FFT-Adaptive",
	"Dierckx-5",
	"GSS",
	"Savitzky-Golay-Fixed",
	"Savitzky-Golay-Adaptive",
	"SG-Package-Fixed",
	"SG-Package-Hybrid",
	"SG-Package-Adaptive",
	"TVRegDiff-Julia",
	"Central-FD",
]

# Step 2: Collect all Python methods that were attempted (from output JSON files)
python_methods_attempted = Set{String}()
predictions_dir = joinpath(@__DIR__, "..", "build", "results", "comprehensive", "predictions")
if isdir(predictions_dir)
	for json_file in readdir(predictions_dir, join=true)
		if endswith(json_file, ".json")
			try
				json_data = JSON3.read(read(json_file, String))
				if haskey(json_data, :methods)
					for (method_name, _) in json_data[:methods]
						if json_data[:methods][method_name][:language] == "Python"
							push!(python_methods_attempted, String(method_name))
						end
					end
				end
			catch e
				# Skip malformed JSON files
			end
		end
	end
end

# Combine all attempted methods
all_methods_attempted = vcat(julia_methods_attempted, collect(python_methods_attempted))

# Step 3: For each (method, ode, noise, order) combination, count successful trials
expected_trials = TRIALS_PER_CONFIG * length(enabled_ode_keys)
orders = collect(0:MAX_DERIV)

# Build a dictionary of actual successful trial counts
success_counts = Dict{Tuple{String, String, Float64, Int}, Int}()
for row in eachrow(df)
	key = (row.method, row.ode_system, row.noise_level, row.deriv_order)
	success_counts[key] = get(success_counts, key, 0) + 1
end

# Step 4: Generate failure report for all attempted methods
failure_records = []
for method in all_methods_attempted
	for order in orders
		for noise_level in NOISE_LEVELS
			# Count successes across all ODEs
			total_successes = 0
			affected_odes = String[]
			for ode_key in enabled_ode_keys
				key = (method, ode_key, noise_level, order)
				successes = get(success_counts, key, 0)
				total_successes += successes
				if successes < TRIALS_PER_CONFIG
					push!(affected_odes, ode_key)
				end
			end

			failures = expected_trials - total_successes
			if failures > 0 || total_successes > 0  # Include if any activity (success or failure)
				push!(failure_records, (
					method = method,
					deriv_order = order,
					noise_level = noise_level,
					total_successful = total_successes,
					total_failures = failures,
					failure_rate = failures / expected_trials,
					affected_odes = join(affected_odes, ", ")
				))
			end
		end
	end
end

# Create DataFrame and sort
failure_report = DataFrame(failure_records)
sort!(failure_report, [:total_failures, :method], rev=[true, false])
CSV.write(joinpath(results_dir, "failure_report.csv"), failure_report)

# Print summary of failures
println("\nFailure Summary:")
critical_failures = filter(row -> row.total_failures > 0, failure_report)
if nrow(critical_failures) > 0
	println("  $(nrow(critical_failures)) method/config combinations have failures")
	top_failures = first(critical_failures, 10)
	for row in eachrow(top_failures)
		println("    $(row.method) (order=$(row.deriv_order), noise=$(row.noise_level)): $(row.total_failures)/$(expected_trials) failures ($(round(row.failure_rate*100, digits=1))%)")
	end
	if nrow(critical_failures) > 10
		println("    ... and $(nrow(critical_failures) - 10) more (see failure_report.csv)")
	end
else
	println("  ✓ No failures detected - all methods completed successfully!")
end

# ===================== One-pass retry for missing configs =====================
println("\nAttempting one-pass retry for missing configs...")
missing_configs = _find_missing_configs(df, enabled_ode_keys, NOISE_LEVELS, TRIALS_PER_CONFIG)
if isempty(missing_configs)
	println("  ✓ No missing (ode, noise, trial) configs detected")
else
	println("  Missing configs: $(length(missing_configs)) - retrying once...")
	orders_retry = collect(0:MAX_DERIV)
	retry_results_all = Vector{NamedTuple}()
	for (ode_key, nl, t) in missing_configs
		try
			append!(retry_results_all, _retry_config_once(ode_key, nl, t, orders_retry, PYTHON_VENV, PYTHON_SCRIPT))
		catch e
			@warn "Retry for $(ode_key), noise=$(nl), trial=$(t) failed" exception=(e,)
		end
	end
	if !isempty(retry_results_all)
		# Append and re-save results and summaries
		df_retry = DataFrame(retry_results_all)
		df = vcat(df, df_retry)
		CSV.write(joinpath(results_dir, "comprehensive_results.csv"), df)

		# Recompute summary and failure report
		summary = combine(groupby(df, [:ode_system, :method, :category, :language, :deriv_order, :noise_level])) do sdf
			(
				mean_rmse = mean(sdf.rmse),
				std_rmse = std(sdf.rmse),
				min_rmse = minimum(sdf.rmse),
				max_rmse = maximum(sdf.rmse),
				mean_mae = mean(sdf.mae),
				mean_nrmse = mean(sdf.nrmse),
				std_nrmse = std(sdf.nrmse),
				min_nrmse = minimum(sdf.nrmse),
				max_nrmse = maximum(sdf.nrmse),
				mean_timing = mean(sdf.timing),
				trials = nrow(sdf),
			)
		end
		CSV.write(joinpath(results_dir, "comprehensive_summary.csv"), summary)

		expected_trials = TRIALS_PER_CONFIG * length(enabled_ode_keys)
		failure_report = combine(groupby(summary, [:method, :deriv_order, :noise_level])) do sdf
			total_trials = sum(sdf.trials)
			failures = expected_trials - total_trials
			(
				total_successful = total_trials,
				total_failures = failures,
				failure_rate = failures / expected_trials,
				affected_odes = join([row.ode_system for row in eachrow(sdf) if row.trials < TRIALS_PER_CONFIG], ", "),
			)
		end
		sort!(failure_report, :total_failures, rev = true)
		CSV.write(joinpath(results_dir, "failure_report.csv"), failure_report)
		println("  Retry complete. Updated failure report written.")
	else
		println("  Retry produced no additional results.")
	end
end

println("\n" * "=" ^ 80)
println("COMPREHENSIVE STUDY COMPLETE")
println("=" ^ 80)
println("\nResults saved to: $(results_dir)")
println("  - comprehensive_results.csv (aggregated metrics)")
println("  - comprehensive_summary.csv (summary statistics)")
println("  - failure_report.csv (methods with missing/failed trials)")
println("  - predictions/ (raw prediction arrays for visualization)")
println("=" ^ 80)
