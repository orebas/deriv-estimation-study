#!/usr/bin/env julia
"""
Sweep through different filter_frac values to optimize Fourier-Interp.
Find the best regularization parameter for balancing smoothness vs accuracy.
"""

using Random
using Printf
include("src/julia_methods.jl")

println("="^70)
println("Fourier-Interp: Filter Fraction Optimization")
println("="^70)

# Generate test data
Random.seed!(42)
x = collect(range(0, 10, length=100))
y_true = @. sin(x) + 0.5*cos(2*x)
noise_level = 0.01
y_noisy = y_true .+ noise_level .* randn(length(x))

# Ground truth at x=5.0
ground_truth = Dict(
    0 => sin(5.0) + 0.5*cos(10.0),
    1 => cos(5.0) - sin(10.0),
    2 => -sin(5.0) - 2*cos(10.0),
    3 => -cos(5.0) + 4*sin(10.0),
    4 => sin(5.0) + 8*cos(10.0),
    5 => cos(5.0) - 16*sin(10.0),
    6 => -sin(5.0) - 32*cos(10.0),
    7 => -cos(5.0) + 64*sin(10.0)
)

println("Data: $(length(x)) points, $(100*noise_level)% noise")
println()

# Fit once (reuse for all filter_frac values)
fourier_fft = fit_fourier(x, y_noisy)

# Test different filter fractions
filter_fracs = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

# Store results for each filter_frac
results = Dict()

for filter_frac in filter_fracs
    errors = Float64[]

    for order in 0:7
        pred = fourier_fft_deriv(fourier_fft, 5.0, order; filter_frac=filter_frac)
        truth = ground_truth[order]
        error = abs(pred - truth)
        push!(errors, error)
    end

    results[filter_frac] = errors
end

# Print comparison table
println("Filter Fraction Comparison (errors at x=5.0)")
println("-"^70)

# Header
@printf("%-12s", "filter_frac")
for order in 0:7
    @printf(" | %9s", "ord-$order")
end
println()
println("-"^70)

# Data rows
for filter_frac in filter_fracs
    @printf("%-12.2f", filter_frac)
    for error in results[filter_frac]
        if error < 1.0
            @printf(" | %9.2e", error)
        elseif error < 1000
            @printf(" | %9.1f", error)
        else
            @printf(" | %9.2e", error)
        end
    end
    println()
end

println("-"^70)
println()

# Find best filter_frac for each order
println("Best filter_frac for each order:")
println("-"^70)
for order in 0:7
    best_frac = nothing
    best_error = Inf

    for filter_frac in filter_fracs
        error = results[filter_frac][order+1]
        if error < best_error
            best_error = error
            best_frac = filter_frac
        end
    end

    @printf("Order %d: filter_frac=%.2f (error=%.2e)\n", order, best_frac, best_error)
end

println("-"^70)
println()

# Compute average rank for each filter_frac (lower is better)
println("Overall ranking (average rank across all orders):")
println("-"^70)
avg_ranks = Dict()

for filter_frac in filter_fracs
    ranks = Int[]
    for order in 0:7
        # Sort all filter_fracs by error for this order
        errors_this_order = [(ff, results[ff][order+1]) for ff in filter_fracs]
        sort!(errors_this_order, by=x->x[2])

        # Find rank of current filter_frac
        rank = findfirst(x -> x[1] == filter_frac, errors_this_order)
        push!(ranks, rank)
    end

    avg_ranks[filter_frac] = mean(ranks)
end

# Sort by average rank
sorted_fracs = sort(collect(avg_ranks), by=x->x[2])

for (filter_frac, avg_rank) in sorted_fracs
    @printf("filter_frac=%.2f: avg rank %.1f\n", filter_frac, avg_rank)
end

println("-"^70)
println()
println("Recommendation: filter_frac=$(sorted_fracs[1][1]) has best overall performance")
println("="^70)
