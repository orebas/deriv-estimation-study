"""
Test boundary handling in the ACTUAL PyNumDiff and integrated methods
"""

using Plots
using Statistics
using Random
using Printf

gr()

# Import our actual methods
include("src/julia_methods_integrated.jl")

# Generate test signal
n = 201
t = collect(range(0, 1, length=n))
dt = t[2] - t[1]

# True signal and derivatives
y_true = sin.(2π .* t)
dy_true = 2π .* cos.(2π .* t)

# Add noise
Random.seed!(42)
noise_level = 1e-3
y_noisy = y_true .+ noise_level .* randn(n)

println("="^80)
println("TESTING ACTUAL PYNUMDIFF BOUNDARY HANDLING")
println("="^80)
println()

# Test PyNumDiff methods
pynumdiff_methods = [
    "smooth_acceleration",
    "finite_difference",
    "spline",
    "trend_filtered",
    "kalman",
    "kernel_regression",
    "chebychev"
]

# Storage for results
boundary_errors = Dict()
interior_errors = Dict()

# Define regions
inner_80 = 21:181  # Middle 80%
boundary_10 = vcat(1:10, 192:201)  # 10% at each end
boundary_20 = vcat(1:20, 182:201)  # 20% at each end

println("Running PyNumDiff methods...")
for method in pynumdiff_methods
    try
        # Use the actual integrated method
        result = compute_derivative(
            t, y_noisy, method, 1;  # Order 1 for derivative
            alpha=0.01,  # Regularization
            options=Dict()
        )

        if haskey(result["predictions"], 1)
            dy_est = result["predictions"][1]

            # Compute errors by region
            rmse_interior = sqrt(mean((dy_est[inner_80] - dy_true[inner_80]).^2))
            rmse_boundary = sqrt(mean((dy_est[boundary_20] - dy_true[boundary_20]).^2))

            boundary_errors[method] = rmse_boundary
            interior_errors[method] = rmse_interior

            @printf("%-20s | Interior: %.4f | Boundary: %.4f | Ratio: %.1fx\n",
                    method, rmse_interior, rmse_boundary, rmse_boundary/rmse_interior)
        else
            println("$method: No first derivative available")
        end
    catch e
        println("$method: Failed - $e")
    end
end

println("\nRunning Julia native methods...")
julia_methods = [
    "savitzky_golay",
    "whittaker_smooth",
    "central_diff",
    "holoborodko",
    "spectral_derivative"
]

for method in julia_methods
    try
        result = compute_derivative(
            t, y_noisy, method, 1;
            alpha=0.01,
            options=Dict()
        )

        if haskey(result["predictions"], 1)
            dy_est = result["predictions"][1]

            rmse_interior = sqrt(mean((dy_est[inner_80] - dy_true[inner_80]).^2))
            rmse_boundary = sqrt(mean((dy_est[boundary_20] - dy_true[boundary_20]).^2))

            boundary_errors[method] = rmse_boundary
            interior_errors[method] = rmse_interior

            @printf("%-20s | Interior: %.4f | Boundary: %.4f | Ratio: %.1fx\n",
                    method, rmse_interior, rmse_boundary, rmse_boundary/rmse_interior)
        end
    catch e
        println("$method: Failed - $e")
    end
end

# Create visualization of the worst offenders
println("\nVisualizing methods with worst boundary effects...")

# Find worst boundary performers
sorted_by_ratio = sort(
    [(m, boundary_errors[m]/interior_errors[m]) for m in keys(boundary_errors)],
    by = x -> x[2],
    rev = true
)

# Plot the worst 3
p = plot(layout=(2,2), size=(1200, 800))

# Get derivatives for worst performers
worst_methods = sorted_by_ratio[1:min(3, length(sorted_by_ratio))]

for (idx, (method, ratio)) in enumerate(worst_methods)
    # Recompute to get the derivative
    result = compute_derivative(t, y_noisy, method, 1; alpha=0.01)
    dy_est = result["predictions"][1]

    # Plot in subplot
    plot!(p[idx], t, dy_true, label="True", linewidth=3, color=:black,
          title="$method (boundary error $(round(ratio, digits=1))x worse)")
    plot!(p[idx], t, dy_est, label="Estimated", linewidth=2, color=:red, alpha=0.7)

    # Mark boundary regions
    vspan!(p[idx], [0, t[20]], color=:gray, alpha=0.2, label=nothing)
    vspan!(p[idx], [t[181], 1], color=:gray, alpha=0.2, label=nothing)

    xlabel!(p[idx], "t")
    ylabel!(p[idx], "dy/dt")
end

# Plot error map for all methods
errors_all = Float64[]
method_names = String[]
positions = Float64[]

for (method, _) in sorted_by_ratio[1:min(6, length(sorted_by_ratio))]
    result = compute_derivative(t, y_noisy, method, 1; alpha=0.01)
    dy_est = result["predictions"][1]
    errors = abs.(dy_est - dy_true)

    append!(errors_all, errors)
    append!(method_names, fill(method, length(errors)))
    append!(positions, t)
end

# Create error heatmap-like plot in 4th subplot
plot!(p[4], title="Error Distribution (6 worst methods)", legend=:topright)
for (i, (method, _)) in enumerate(sorted_by_ratio[1:min(6, length(sorted_by_ratio))])
    result = compute_derivative(t, y_noisy, method, 1; alpha=0.01)
    dy_est = result["predictions"][1]
    errors = abs.(dy_est - dy_true)

    plot!(p[4], t, errors .+ (i-1)*0.1, label=method, linewidth=1.5, alpha=0.7)
end
vline!(p[4], [t[20], t[181]], color=:gray, linestyle=:dash, label="boundary", alpha=0.5)
xlabel!(p[4], "t")
ylabel!(p[4], "Error (offset for visibility)")

savefig(p, "pynumdiff_boundary_analysis.png")

println("\n" * "="^80)
println("SUMMARY")
println("="^80)
println("\nMethods with boundary/interior error ratio > 10x:")
for (method, ratio) in sorted_by_ratio
    if ratio > 10
        println("  - $method: $(round(ratio, digits=1))x worse at boundaries")
    end
end

println("\nMethods with good boundary handling (ratio < 2x):")
for (method, ratio) in sorted_by_ratio
    if ratio < 2
        println("  - $method: $(round(ratio, digits=1))x")
    end
end