#!/usr/bin/env julia

# Minimal repro of two GP issues we are seeing in this repo:
# 1) AD through GaussianProcesses.jl predictor closure for derivatives > 0
# 2) Positive-definiteness/Cholesky failures on analytic SE GP with tiny noise

using Random, Statistics, LinearAlgebra, Printf
using GaussianProcesses, Optim, LineSearches
using ForwardDiff, TaylorDiff

Random.seed!(123)

function make_data(n::Int; noiseσ::Float64 = 0.0)
	x = collect(range(0.0, 10.0, length = n))
	yclean = @. sin(x) + 0.3cos(0.7x)
	y = yclean .+ noiseσ .* randn(length(x))
	return x, y, yclean
end

function test_ad_through_gp(n::Int)
	println("\n" * "-"^70)
	println("AD test through GaussianProcesses.jl predictor (n=$(n))")
	x, y, yclean = make_data(n)

	# Build GP with small noise and SE kernel
	kern = SEIso(0.0, 0.0)  # log ℓ, log σf
	gp = GP(x, y, MeanZero(), kern, log(1e-4))
	try
		optimize!(gp; method = LBFGS(linesearch = LineSearches.BackTracking()))
	catch e
		@warn "GP optimize! failed, proceeding with current hyperparams" exception=e
	end

	f = function (t)
		μ, _ = predict_f(gp, [t])
		return μ[1]
	end

	t0 = x[cld(n, 2)]
	# Try ForwardDiff and TaylorDiff for orders 0..3
	for order in 0:3
		val_fd = NaN
		val_td = NaN
		err_fd = nothing
		err_td = nothing
		try
			val_fd = (order == 0) ? f(t0) : ForwardDiff.derivative(t -> begin
				;
				f(t);
			end, t0)
		catch e
			err_fd = e
		end
		try
			val_td = (order == 0) ? f(t0) : TaylorDiff.derivative(f, t0, Val(order))
		catch e
			err_td = e
		end
		@printf("order %d: ForwardDiff=%s  TaylorDiff=%s\n",
			order,
			isnan(val_fd) ? string(err_fd) : @sprintf("%.6e", val_fd),
			isnan(val_td) ? string(err_td) : @sprintf("%.6e", val_td))
	end
	println("Note: On our LV baseline this path produced invalid derivatives (Inf/NaN).\n" *
			"This script shows whether AD returns values or errors at a sample point.")
end

function cholesky_with_escalating_jitter!(K::Matrix{Float64})
	dmed = median(diag(K))
	jitter = max(1e-12, 1e-8 * dmed)
	for k in 1:12
		try
			return cholesky(Symmetric(K)), jitter, true
		catch
			@inbounds for i in 1:size(K, 1)
				K[i, i] += jitter
			end
			jitter *= 10
		end
	end
	return cholesky(Symmetric(K); check = false), jitter, false
end

function test_pd_on_analytic_se(n::Int)
	println("\n" * "-"^70)
	println("PD (Cholesky) test on analytic SE GP with tiny noise (n=$(n))")
	x, y, _ = make_data(n)
	y = y .- mean(y)
	# Scale x for conditioning
	xμ, xσ = mean(x), std(x)
	xσ = xσ == 0 ? 1.0 : xσ
	xs = (x .- xμ) ./ xσ

	# MLE on SE params (log ℓ, log σf, log σn)
	function nll(p)
		ℓ = exp(p[1]);
		σf = exp(p[2]);
		σn = max(exp(p[3]), 1e-6)
		# Build K
		D2 = (repeat(xs, 1, length(xs)) .- repeat(xs', length(xs), 1)) .^ 2
		K = (σf^2) .* exp.(-0.5 .* (D2 ./ (ℓ^2)))
		@inbounds for i in 1:length(xs)
			K[i, i] += σn^2 + 1e-10
		end
		F = cholesky(Symmetric(K))
		α = F \ y
		return 0.5 * dot(y, α) + sum(log, diag(F.U)) + 0.5 * length(xs) * log(2π)
	end

	p0 = [log(std(xs)/8), log(std(y)+eps()), log(std(y)/100 + 1e-6)]
	res = Optim.optimize(nll, p0, LBFGS(); autodiff = :forward)
	@printf("MLE status: %s, f=%.6f\n", string(Optim.termination_status(res)), Optim.minimum(res))
	p̂ = Optim.minimizer(res)
	ℓ̂, σf̂, σn̂ = exp.(p̂)
	@printf("ℓ=%.3e  σf=%.3e  σn=%.3e\n", ℓ̂, σf̂, σn̂)

	# Rebuild K with fitted params and try Cholesky with escalating jitter
	D2 = (repeat(xs, 1, length(xs)) .- repeat(xs', length(xs), 1)) .^ 2
	K = (σf̂^2) .* exp.(-0.5 .* (D2 ./ (ℓ̂^2)))
	@inbounds for i in 1:length(xs)
		K[i, i] += σn̂^2
	end
	try
		cholesky(Symmetric(K))
		println("Cholesky(K) OK without jitter")
	catch
		F, lastjit, ok = cholesky_with_escalating_jitter!(K)
		@printf("Cholesky needed jitter. final_jitter≈%.3e  ok=%s\n", lastjit, string(ok))
	end
end

function main()
	n = try
		parse(Int, get(ENV, "NPTS", "51"))
	catch
		; 51
	end
	test_ad_through_gp(n)
	test_pd_on_analytic_se(n)
end

main()

# Run with:
#   julia --startup-file=no --project src/repro_gp_issues.jl
# Adjust NPTS via env if desired, e.g. NPTS=51 or 201.


