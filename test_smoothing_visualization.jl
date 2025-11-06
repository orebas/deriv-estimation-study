"""
Visualize how different smoothing methods work (or don't)
and explore Kalman filter tuning options
"""

using ApproxFun
using Statistics
using Printf
using Interpolations
using Plots
using FFTW
using Random
using LinearAlgebra

gr()  # Use GR backend for speed

# Test signal
n = 201
t = collect(range(0, 1, length=n))
ω = 2π
y_true = sin.(ω .* t)
noise_level = 1e-3
Random.seed!(42)  # Reproducible
y_noisy = y_true .+ noise_level .* randn(n)

println("="^80)
println("Smoothing Methods Visualization")
println("="^80)

# Different smoothing methods
function moving_average(y, window=11)
    n = length(y)
    y_smooth = similar(y)
    half_win = window ÷ 2
    for i in 1:n
        i_start = max(1, i - half_win)
        i_end = min(n, i + half_win)
        y_smooth[i] = mean(y[i_start:i_end])
    end
    return y_smooth
end

function gaussian_smooth(y, σ=2.0)
    n = length(y)
    y_smooth = similar(y)
    window = ceil(Int, 4*σ)  # ±2σ window
    for i in 1:n
        weights = Float64[]
        values = Float64[]
        for j in max(1, i-window):min(n, i+window)
            w = exp(-0.5 * ((i - j) / σ)^2)
            push!(weights, w)
            push!(values, y[j])
        end
        weights ./= sum(weights)
        y_smooth[i] = sum(weights .* values)
    end
    return y_smooth
end

function kalman_smooth(y, α=0.98; process_noise=1e-6, meas_noise=1e-3)
    """
    Simple scalar Kalman filter
    α is like a smoothing parameter (higher = smoother but more lag)
    process_noise and meas_noise control the filter behavior
    """
    n = length(y)
    y_smooth = zeros(n)

    # Forward pass
    x = y[1]  # Initial state
    P = 1.0   # Initial covariance

    for i in 1:n
        # Predict
        x_pred = α * x
        P_pred = α^2 * P + process_noise

        # Update
        K = P_pred / (P_pred + meas_noise)  # Kalman gain
        x = x_pred + K * (y[i] - x_pred)
        P = (1 - K) * P_pred

        y_smooth[i] = x
    end

    # Backward pass for smoothing (RTS smoother)
    x_smooth = zeros(n)
    x_smooth[end] = y_smooth[end]

    for i in (n-1):-1:1
        # Simplified backward smoothing
        x_smooth[i] = 0.5 * (y_smooth[i] + x_smooth[i+1])
    end

    return x_smooth
end

# Better Kalman with state space model for derivatives
function kalman_smooth_advanced(y, dt=1.0/(length(y)-1);
                               process_std=1e-4, meas_std=1e-3, order=2)
    """
    Advanced Kalman that tracks position and derivatives
    order: 1 = position+velocity, 2 = position+velocity+acceleration
    """
    n = length(y)

    # State dimension
    state_dim = order + 1

    # State transition matrix (constant velocity or acceleration model)
    F = zeros(state_dim, state_dim)
    for i in 1:state_dim
        F[i,i] = 1.0
        if i < state_dim
            F[i,i+1] = dt
            if i < state_dim - 1 && order > 1
                F[i,i+2] = 0.5*dt^2  # For acceleration
            end
        end
    end

    # Measurement matrix (we only observe position)
    H = zeros(1, state_dim)
    H[1,1] = 1.0

    # Process noise
    Q = diagm([(process_std * dt^i)^2 for i in 0:order])

    # Measurement noise
    R = [meas_std^2]

    # Initialize
    x = zeros(state_dim, n)  # States
    x[:,1] = [y[1]; zeros(order)]  # Initial state
    P = Matrix(1.0I, state_dim, state_dim)  # Initial covariance

    # Forward filter
    for i in 2:n
        # Predict
        x_pred = F * x[:,i-1]
        P_pred = F * P * F' + Q

        # Update
        y_pred = H * x_pred
        S = H * P_pred * H' + R
        K = P_pred * H' / S[1]  # Kalman gain

        x[:,i] = x_pred + K * (y[i] - y_pred[1])
        P = (I - K * H) * P_pred
    end

    # Extract smoothed position
    return x[1,:]
end

# PLOT 1: Signal comparison
println("\nGenerating plots...")
p1 = plot(t, y_true, label="True", linewidth=2, alpha=0.8, title="Signal Comparison")
plot!(t, y_noisy, label="Noisy", alpha=0.3, markersize=1)
plot!(t, moving_average(y_noisy), label="Moving Avg", linewidth=1.5)
plot!(t, gaussian_smooth(y_noisy), label="Gaussian", linewidth=1.5)
plot!(t, kalman_smooth(y_noisy, 0.98), label="Kalman (α=0.98)", linewidth=1.5)
plot!(t, kalman_smooth(y_noisy, 0.90), label="Kalman (α=0.90)", linewidth=1.5)
xlabel!("t")
ylabel!("y")

# PLOT 2: Zoomed in view to see roughness
t_zoom = 41:81  # Zoom into middle section
p2 = plot(t[t_zoom], y_true[t_zoom], label="True", linewidth=3,
          title="Zoomed View (Roughness Visible)", marker=:circle, markersize=2)
plot!(t[t_zoom], y_noisy[t_zoom], label="Noisy", alpha=0.5, marker=:x, markersize=3)
plot!(t[t_zoom], moving_average(y_noisy)[t_zoom], label="Moving Avg",
      linewidth=2, marker=:square, markersize=2)
plot!(t[t_zoom], kalman_smooth(y_noisy, 0.98)[t_zoom], label="Kalman (α=0.98)",
      linewidth=2, marker=:diamond, markersize=2)
xlabel!("t")
ylabel!("y")

# PLOT 3: First differences (roughness indicator)
p3 = plot(title="First Differences (Roughness)", legend=:topright)
for (name, y_smooth, style) in [
    ("Moving Avg", moving_average(y_noisy), :solid),
    ("Gaussian", gaussian_smooth(y_noisy), :dash),
    ("Kalman 0.98", kalman_smooth(y_noisy, 0.98), :dot),
    ("Kalman 0.90", kalman_smooth(y_noisy, 0.90), :dashdot)
]
    plot!(diff(y_smooth), label=name, linestyle=style, alpha=0.7)
end
plot!(diff(y_true), label="True (smooth)", linewidth=2, color=:black)
xlabel!("Index")
ylabel!("Δy")

# PLOT 4: Frequency spectrum
p4 = plot(title="Frequency Spectrum", yaxis=:log, legend=:topright)
freqs = fftfreq(n, 1.0/(t[2]-t[1]))[1:div(n,2)]
for (name, y_smooth, style) in [
    ("Noisy", y_noisy, :solid),
    ("Moving Avg", moving_average(y_noisy), :dash),
    ("Kalman 0.98", kalman_smooth(y_noisy, 0.98), :dot),
    ("Kalman 0.90", kalman_smooth(y_noisy, 0.90), :dashdot)
]
    spectrum = abs.(fft(y_smooth))[1:div(n,2)]
    plot!(freqs[2:end], spectrum[2:end], label=name, linestyle=style, linewidth=1.5)
end
xlabel!("Frequency")
ylabel!("Magnitude")

# Combine plots
p_main = plot(p1, p2, p3, p4, layout=(2,2), size=(1000,800))
savefig(p_main, "smoothing_comparison.png")
println("Saved: smoothing_comparison.png")

# PLOT 5: Kalman parameter exploration
println("\nExploring Kalman parameters...")

alphas = [0.8, 0.85, 0.9, 0.95, 0.98, 0.99]
p5 = plot(title="Kalman Filter: Effect of α Parameter", size=(800,600))
plot!(t, y_true, label="True", linewidth=3, color=:black)

for (i, α) in enumerate(alphas)
    y_kalman = kalman_smooth(y_noisy, α)

    # Compute smoothness metric
    roughness = std(diff(y_kalman))
    signal_error = sqrt(mean((y_kalman - y_true).^2))

    label_text = @sprintf("α=%.2f (rough=%.1e, err=%.1e)", α, roughness, signal_error)
    plot!(t, y_kalman, label=label_text, linewidth=1.5, alpha=0.7)
end

xlabel!("t")
ylabel!("y")
savefig(p5, "kalman_parameter_sweep.png")
println("Saved: kalman_parameter_sweep.png")

# PLOT 6: Advanced Kalman comparison
println("\nTesting advanced Kalman filters...")

p6 = plot(title="Advanced Kalman Models", size=(800,600))
plot!(t, y_true, label="True", linewidth=3, color=:black)

# Simple Kalman
y_simple = kalman_smooth(y_noisy, 0.95)
plot!(t, y_simple, label="Simple Kalman", linewidth=2)

# Advanced Kalman with different orders
for order in [1, 2]
    y_advanced = kalman_smooth_advanced(y_noisy, order=order,
                                       process_std=1e-4, meas_std=noise_level)
    plot!(t, y_advanced, label="Advanced Kalman (order $order)", linewidth=2)
end

xlabel!("t")
ylabel!("y")
savefig(p6, "kalman_advanced.png")
println("Saved: kalman_advanced.png")

# PLOT 7: ApproxFun coefficient decay visualization
println("\nVisualizing Chebyshev coefficient decay...")

methods_to_test = [
    ("Moving Average", moving_average(y_noisy)),
    ("Kalman α=0.98", kalman_smooth(y_noisy, 0.98)),
    ("Kalman α=0.90", kalman_smooth(y_noisy, 0.90)),
    ("Advanced Kalman", kalman_smooth_advanced(y_noisy, order=2))
]

p7 = plot(title="Chebyshev Coefficient Decay", yaxis=:log, size=(800,600))

for (name, y_smooth) in methods_to_test
    # Create interpolator and fit with ApproxFun
    itp = linear_interpolation(t, y_smooth, extrapolation_bc=Flat())

    try
        # Get first 100 coefficients to see decay pattern
        f = Fun(x -> itp(x), 0..1)
        coeffs = coefficients(f)

        # Plot first 50 coefficients
        n_plot = min(50, length(coeffs))
        plot!(1:n_plot, abs.(coeffs[1:n_plot]) .+ 1e-16,
              label=name, marker=:circle, markersize=2, linewidth=1.5)

    catch e
        println("  $name: Failed to compute coefficients")
    end
end

# Add reference line for good exponential decay
n_ref = 50
good_decay = 1.0 * exp.(-0.5 * (0:n_ref-1))
plot!(1:n_ref, good_decay, label="Good decay (exp(-0.5n))",
      linestyle=:dash, linewidth=2, color=:green)

xlabel!("Coefficient Index")
ylabel!("|Coefficient|")
savefig(p7, "coefficient_decay.png")
println("Saved: coefficient_decay.png")

# PLOT 8: The killer plot - second derivatives
println("\nComputing second derivatives to show the problem...")

p8 = plot(title="Second Derivatives (Curvature)", size=(800,600))

# True second derivative
d2_true = -(ω^2) * sin.(ω * t)
plot!(t[2:end-1], d2_true[2:end-1], label="True", linewidth=3, color=:black)

# Compute numerical second derivatives
for (name, y_smooth, style) in [
    ("Moving Avg", moving_average(y_noisy), :solid),
    ("Kalman 0.98", kalman_smooth(y_noisy, 0.98), :dash),
    ("Advanced Kalman", kalman_smooth_advanced(y_noisy, order=2), :dot)
]
    # Numerical second derivative
    d2_numerical = zeros(n-2)
    for i in 2:(n-1)
        d2_numerical[i-1] = (y_smooth[i-1] - 2*y_smooth[i] + y_smooth[i+1]) / (t[2]-t[1])^2
    end

    plot!(t[2:end-1], d2_numerical, label=name, linestyle=style,
          linewidth=1.5, alpha=0.7)
end

xlabel!("t")
ylabel!("d²y/dt²")
savefig(p8, "second_derivatives.png")
println("Saved: second_derivatives.png")

# Final analysis
println("\n" * "="^80)
println("QUANTITATIVE ANALYSIS")
println("="^80)

println("\nMethod Comparison (lower is better):")
println("-"^60)
println("Method               | Roughness | RMSE to True | Max |d²y/dt²|")
println("-"^60)

test_methods = [
    ("Raw Noisy", y_noisy),
    ("Moving Avg", moving_average(y_noisy)),
    ("Gaussian σ=2", gaussian_smooth(y_noisy, 2.0)),
    ("Gaussian σ=4", gaussian_smooth(y_noisy, 4.0)),
    ("Kalman α=0.98", kalman_smooth(y_noisy, 0.98)),
    ("Kalman α=0.95", kalman_smooth(y_noisy, 0.95)),
    ("Kalman α=0.90", kalman_smooth(y_noisy, 0.90)),
    ("Kalman α=0.85", kalman_smooth(y_noisy, 0.85)),
    ("Adv. Kalman ord=1", kalman_smooth_advanced(y_noisy, order=1)),
    ("Adv. Kalman ord=2", kalman_smooth_advanced(y_noisy, order=2))
]

best_roughness = Inf
best_rmse = Inf
best_method = ""

for (name, y_smooth) in test_methods
    roughness = std(diff(y_smooth))
    rmse = sqrt(mean((y_smooth - y_true).^2))

    # Compute max second derivative
    d2_max = 0.0
    for i in 2:(n-1)
        d2 = abs(y_smooth[i-1] - 2*y_smooth[i] + y_smooth[i+1]) / (t[2]-t[1])^2
        d2_max = max(d2_max, d2)
    end

    @printf("%-20s | %.3e  | %.3e    | %.1f\n", name, roughness, rmse, d2_max)

    # Track best trade-off
    score = roughness + 10*rmse  # Weight accuracy more
    if score < best_roughness + 10*best_rmse
        global best_roughness = roughness
        global best_rmse = rmse
        global best_method = name
    end
end

println("\nBest trade-off: $best_method")
println("  Roughness: $(round(best_roughness, sigdigits=3))")
println("  RMSE: $(round(best_rmse, sigdigits=3))")

# Test ApproxFun on the best method
println("\nTesting ApproxFun on best method...")
best_y = test_methods[findfirst(m -> m[1] == best_method, test_methods)][2]
itp_best = linear_interpolation(t, best_y, extrapolation_bc=Flat())

try
    f_best = Fun(x -> itp_best(x), 0..1)
    n_coeffs = length(coefficients(f_best))
    println("ApproxFun coefficients needed: $n_coeffs")

    if n_coeffs < 100
        println("SUCCESS! This might be smooth enough for spectral methods!")
    elseif n_coeffs < 1000
        println("Marginal - might work for low-order derivatives")
    else
        println("FAILURE - still too rough for spectral methods")
    end
catch e
    println("ApproxFun failed completely!")
end

println("\n" * "="^80)
println("CONCLUSION")
println("="^80)

println("""
The visualizations show:

1. Simple smoothing (moving average, Gaussian) barely reduces roughness
   - First differences still oscillate wildly
   - Second derivatives are catastrophically noisy

2. Kalman with high α (0.98-0.99) oversmooths
   - Loses signal features
   - Introduces systematic bias

3. Kalman with medium α (0.85-0.95) is a compromise
   - Still rough at the microscopic level
   - Better preserves signal

4. Advanced Kalman with derivative tracking helps slightly
   - But still not smooth enough for spectral methods

5. The fundamental issue: These methods optimize for different goals
   - PyNumDiff: Numerical stability in finite differences
   - ApproxFun: Analytical smoothness for spectral methods
   - These goals are incompatible!

Check the generated plots to see the visual evidence.
""")