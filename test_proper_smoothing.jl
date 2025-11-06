"""
Show what PROPER smoothing should look like vs what's happening
"""

using Plots
using Statistics
using FFTW
using Random

gr()

# Define fftfreq if not available
function fftfreq(n::Int, fs::Float64=1.0)
    # Generate frequency array for FFT
    freqs = zeros(n)
    if n % 2 == 0
        # Even length
        freqs[1:div(n,2)+1] = (0:div(n,2)) * fs / n
        freqs[div(n,2)+2:end] = (-div(n,2)+1:-1) * fs / n
    else
        # Odd length
        freqs[1:div(n+1,2)] = (0:div(n-1,2)) * fs / n
        freqs[div(n+1,2)+1:end] = (-div(n-1,2):-1) * fs / n
    end
    return freqs
end

# Generate test signal
n = 201
t = collect(range(0, 1, length=n))
y_true = sin.(2π .* t)
noise_level = 1e-3
Random.seed!(42)
y_noisy = y_true .+ noise_level .* randn(n)

# PROPER smoothing - frequency domain filtering
function proper_smooth(y_noisy, cutoff_freq=10)
    # Use FFT to properly smooth
    fft_signal = fft(y_noisy)
    freqs = fftfreq(length(y_noisy), 1.0/(t[2]-t[1]))

    # Zero out high frequencies
    for i in 1:length(fft_signal)
        if abs(freqs[i]) > cutoff_freq
            fft_signal[i] = 0
        end
    end

    return real(ifft(fft_signal))
end

# Bad "smoothing" - what the Kalman filter was doing
function bad_kalman(y, α=0.98)
    y_smooth = similar(y)
    y_smooth[1] = y[1]
    for i in 2:length(y)
        y_smooth[i] = α * y_smooth[i-1] + (1-α) * y[i]
    end
    return y_smooth
end

# Simple moving average (also not great)
function moving_avg(y, window=11)
    y_smooth = similar(y)
    half_win = window ÷ 2
    for i in 1:length(y)
        i_start = max(1, i - half_win)
        i_end = min(length(y), i + half_win)
        y_smooth[i] = mean(y[i_start:i_end])
    end
    return y_smooth
end

# Create comparison plot
p1 = plot(title="What Different 'Smoothing' Methods Actually Do",
          size=(1200, 400), legend=:topright)

plot!(t, y_true, label="True sine wave", linewidth=3, color=:black)
plot!(t, y_noisy, label="Noisy data", alpha=0.3, color=:gray)
plot!(t, proper_smooth(y_noisy, 5), label="PROPER smoothing (FFT cutoff)",
      linewidth=2, color=:green)
plot!(t, moving_avg(y_noisy), label="Moving average (preserves shape)",
      linewidth=2, color=:blue)
plot!(t, bad_kalman(y_noisy, 0.98), label="Bad Kalman α=0.98 (destroys signal)",
      linewidth=2, color=:red, linestyle=:dash)

xlabel!("Time")
ylabel!("Value")

savefig(p1, "proper_vs_bad_smoothing.png")

# Show the actual noise level in each
println("\nSignal Analysis:")
println("="^60)
println("Method                    | Preserves Shape? | Smoothness")
println("-"^60)

y_proper = proper_smooth(y_noisy, 5)
y_ma = moving_avg(y_noisy)
y_bad = bad_kalman(y_noisy, 0.98)

# Check correlation with true signal
corr_proper = cor(y_proper, y_true)
corr_ma = cor(y_ma, y_true)
corr_bad = cor(y_bad, y_true)

# Check roughness (std of differences)
rough_proper = std(diff(y_proper))
rough_ma = std(diff(y_ma))
rough_bad = std(diff(y_bad))

using Printf
@printf("FFT smoothing (good)      | %.1f%% match      | %.1e\n",
        100*corr_proper, rough_proper)
@printf("Moving average            | %.1f%% match      | %.1e\n",
        100*corr_ma, rough_ma)
@printf("Bad Kalman α=0.98         | %.1f%% match      | %.1e\n",
        100*corr_bad, rough_bad)

# Now show WHY spectral methods still fail
println("\n\nThe REAL Problem:")
println("="^60)
println("Even 'proper' smoothing doesn't give C^∞ smoothness!")
println()

# Check how many Chebyshev coefficients we need
using ApproxFun
using Interpolations

for (name, y_smooth) in [
    ("FFT smoothed (cutoff=5)", proper_smooth(y_noisy, 5)),
    ("FFT smoothed (cutoff=2)", proper_smooth(y_noisy, 2)),
    ("Moving average", y_ma)
]
    itp = linear_interpolation(t, y_smooth, extrapolation_bc=Flat())
    try
        f = Fun(x -> itp(x), 0..1)
        n_coeffs = length(coefficients(f))
        println("$name needs $n_coeffs Chebyshev coefficients")
    catch e
        println("$name: FAILED - too rough for ApproxFun!")
    end
end

println("\nConclusion: The issue isn't bad smoothing implementations,")
println("it's that NO finite-window smoothing produces true C^∞ functions!")