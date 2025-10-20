#!/usr/bin/env julia

include("ground_truth.jl")
include("julia_methods.jl")

using Printf
using Statistics

function rmse(a::AbstractVector, b::AbstractVector)
	valid = .!isnan.(a) .& .!isinf.(a) .& .!isnan.(b) .& .!isinf.(b)
	if sum(valid) == 0
		return Inf
	end
	return sqrt(mean((a[valid] .- b[valid]) .^ 2))
end

println("="^70)
println("GP AD TEST (TaylorDiff vs ForwardDiff) on LV, noiseless")
println("="^70)

sys = lotka_volterra_system()
times = collect(range(sys.tspan[1], sys.tspan[2], length = 51))
truth = generate_ground_truth(sys, times, 3)

y_true = truth[:obs][1][0]

f_gp = fit_gp(times, y_true; kernel = :SE, optimize = true)

orders = 0:3
for n in orders
	preds_fd = Vector{Float64}(undef, length(times))
	preds_td = Vector{Float64}(undef, length(times))
	for (i, t) in enumerate(times)
		try
			preds_fd[i] = nth_deriv_at(f_gp, n, t)
		catch
			preds_fd[i] = NaN
		end
		try
			if n == 0
				preds_td[i] = f_gp(t)
			else
				preds_td[i] = nth_deriv_taylor(f_gp, n, t)
			end
		catch
			preds_td[i] = NaN
		end
	end
	gt = truth[:obs][1][n]
	r_fd = rmse(preds_fd, gt)
	r_td = rmse(preds_td, gt)
	println(@sprintf("Order %d:  ForwardDiff RMSE = %.6e   TaylorDiff RMSE = %.6e", n, r_fd, r_td))
end

println("\nDone.")


