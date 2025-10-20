"""
Noise Models for High-Order Derivative Study

Provides different noise models with easy switching.
"""

using Random
using Statistics

"""
Noise model types available
"""
@enum NoiseModel begin
    ConstantGaussian    # σ = noise_level * std(signal) [DEFAULT]
    Proportional        # y_noisy = y * (1 + ε), ε ~ N(0, noise_level^2)
    Heteroscedastic     # σ_i = noise_level * (|y_i| + baseline)
end


"""
    add_noise(signal, noise_level, rng; model=ConstantGaussian)

Add noise to a signal according to the specified model.

Arguments:
- signal: Clean signal data (Vector{Float64})
- noise_level: Noise magnitude (e.g., 0.01 for 1%)
- rng: Random number generator for reproducibility
- model: NoiseModel enum value (default: ConstantGaussian)

Returns:
- noisy_signal: Signal with added noise
"""
function add_noise(signal::AbstractVector{T},
                   noise_level::Float64,
                   rng::AbstractRNG;
                   model::NoiseModel=ConstantGaussian) where T<:Real

    n = length(signal)

    if model == ConstantGaussian
        # Additive white Gaussian noise scaled by signal std
        σ = noise_level * std(signal)
        noise = σ .* randn(rng, n)
        return signal .+ noise

    elseif model == Proportional
        # Multiplicative noise: y_noisy = y * (1 + ε)
        ε = noise_level .* randn(rng, n)
        return signal .* (1 .+ ε)

    elseif model == Heteroscedastic
        # Signal-dependent noise with baseline to avoid zeros
        σ_base = noise_level * median(abs.(signal))
        σ_local = @. noise_level * (abs(signal) + 0.1 * σ_base)
        noise = σ_local .* randn(rng, n)
        return signal .+ noise

    else
        error("Unknown noise model: $model")
    end
end


"""
    add_noise_to_data(data, noise_level, rng; model=ConstantGaussian)

Add noise to all observables in a ground truth data structure.

Modifies data[:obs][obs_idx][0] (the 0-th order "derivative" = signal itself).
Leaves true derivatives untouched for error computation.

Arguments:
- data: Ground truth data from generate_ground_truth()
- noise_level: Noise magnitude
- rng: Random number generator
- model: NoiseModel enum

Returns:
- noisy_data: Copy of data with noise added to observables
"""
function add_noise_to_data(data::Dict,
                          noise_level::Float64,
                          rng::AbstractRNG;
                          model::NoiseModel=ConstantGaussian)

    noisy_data = deepcopy(data)

    for (obs_idx, obs_dict) in data[:obs]
        # Add noise only to the observable itself (order 0)
        clean_signal = obs_dict[0]
        noisy_signal = add_noise(clean_signal, noise_level, rng; model=model)
        noisy_data[:obs][obs_idx][0] = noisy_signal
    end

    return noisy_data
end


"""
    estimate_snr(clean, noisy)

Estimate signal-to-noise ratio in dB.

SNR = 10 * log10(Var(signal) / Var(noise))
"""
function estimate_snr(clean::AbstractVector, noisy::AbstractVector)
    noise = noisy .- clean
    signal_power = var(clean)
    noise_power = var(noise)

    if noise_power == 0.0
        return Inf
    end

    return 10 * log10(signal_power / noise_power)
end


# Test function
function test_noise_models()
    println("=" ^ 70)
    println("TESTING NOISE MODELS")
    println("=" ^ 70)

    # Create a simple test signal
    rng = MersenneTwister(12345)
    t = range(0, 10, length=101)
    signal = sin.(2π .* 0.5 .* t) .+ 0.5

    noise_level = 0.05  # 5%

    println("\nTesting noise models with 5% noise:")
    println("Original signal: mean=$(round(mean(signal), digits=3)), std=$(round(std(signal), digits=3))")

    # Test each model
    for model in instances(NoiseModel)
        println("\n$model:")
        rng = MersenneTwister(12345)  # Reset RNG for consistency
        noisy = add_noise(signal, noise_level, rng; model=model)

        snr_db = estimate_snr(signal, noisy)
        println("  Noisy signal: mean=$(round(mean(noisy), digits=3)), std=$(round(std(noisy), digits=3))")
        println("  SNR: $(round(snr_db, digits=2)) dB")
        println("  Max absolute error: $(round(maximum(abs.(noisy .- signal)), digits=4))")
    end

    println("\n✓ Noise model tests complete!")
end

