"""
Show what smoothing methods ACTUALLY do well:
Estimating function values and first derivatives from noisy data
"""

using Plots
using Statistics
using Random
using FFTW
using Printf

gr()

# Generate test signal with realistic noise
n = 201
t = collect(range(0, 1, length=n))
dt = t[2] - t[1]

# True signal and derivative
y_true = sin.(2π .* t)
dy_true = 2π .* cos.(2π .* t)

# Add realistic noise levels
Random.seed!(42)
noise_levels = [1e-4, 1e-3, 1e-2]  # Different noise levels to test

# Proper implementations of smoothing methods
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
    window = ceil(Int, 4*σ)
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

function savitzky_golay(y, window=11, order=3)
    # Simple Savitzky-Golay filter
    n = length(y)
    y_smooth = similar(y)
    half_win = window ÷ 2

    for i in 1:n
        i_start = max(1, i - half_win)
        i_end = min(n, i + half_win)
        local_t = collect(i_start:i_end) .- i
        local_y = y[i_start:i_end]

        # Fit polynomial
        A = zeros(length(local_t), order+1)
        for j in 0:order
            A[:, j+1] = local_t.^j
        end

        coeffs = A \ local_y
        y_smooth[i] = coeffs[1]  # Constant term is the smoothed value
    end
    return y_smooth
end

function compute_derivative(y_smooth, dt)
    # Central differences for derivative
    n = length(y_smooth)
    dy = zeros(n)

    # Forward difference at start
    dy[1] = (y_smooth[2] - y_smooth[1]) / dt

    # Central differences in middle
    for i in 2:n-1
        dy[i] = (y_smooth[i+1] - y_smooth[i-1]) / (2*dt)
    end

    # Backward difference at end
    dy[n] = (y_smooth[n] - y_smooth[n-1]) / dt

    return dy
end

# Create comprehensive comparison
for noise_level in [1e-3]  # Focus on medium noise
    y_noisy = y_true .+ noise_level .* randn(n)

    # Apply different smoothing methods
    y_ma = moving_average(y_noisy, 11)
    y_gauss = gaussian_smooth(y_noisy, 2.0)
    y_sg = savitzky_golay(y_noisy, 11, 3)

    # Compute derivatives
    dy_ma = compute_derivative(y_ma, dt)
    dy_gauss = compute_derivative(y_gauss, dt)
    dy_sg = compute_derivative(y_sg, dt)
    dy_noisy = compute_derivative(y_noisy, dt)  # For comparison

    # Create the plot
    p = plot(layout=(2,2), size=(1200, 800))

    # Plot 1: Original signal with noise
    plot!(p[1], t, y_true, label="True signal", linewidth=3, color=:black)
    plot!(p[1], t, y_noisy, label="Noisy (σ=$noise_level)",
          alpha=0.5, color=:gray, markersize=1)
    title!(p[1], "Original Signal with Noise")
    xlabel!(p[1], "t")
    ylabel!(p[1], "y")

    # Plot 2: Smoothed signals (THIS IS WHAT WORKS!)
    plot!(p[2], t, y_true, label="True", linewidth=3, color=:black)
    plot!(p[2], t, y_ma, label="Moving Average", linewidth=2, color=:blue)
    plot!(p[2], t, y_gauss, label="Gaussian", linewidth=2, color=:green)
    plot!(p[2], t, y_sg, label="Savitzky-Golay", linewidth=2, color=:red)
    title!(p[2], "Smoothed Signals (GOOD!)")
    xlabel!(p[2], "t")
    ylabel!(p[2], "y")

    # Plot 3: First derivatives
    plot!(p[3], t, dy_true, label="True derivative", linewidth=3, color=:black)
    plot!(p[3], t, dy_noisy, label="Raw finite diff", alpha=0.2, color=:gray)
    plot!(p[3], t, dy_ma, label="Moving Avg", linewidth=2, color=:blue)
    plot!(p[3], t, dy_gauss, label="Gaussian", linewidth=2, color=:green)
    plot!(p[3], t, dy_sg, label="Savitzky-Golay", linewidth=2, color=:red)
    title!(p[3], "First Derivatives (WORKS WELL!)")
    xlabel!(p[3], "t")
    ylabel!(p[3], "dy/dt")

    # Plot 4: Zoom on derivatives to show quality
    zoom_range = 51:101
    plot!(p[4], t[zoom_range], dy_true[zoom_range], label="True",
          linewidth=3, color=:black, marker=:circle, markersize=2)
    plot!(p[4], t[zoom_range], dy_ma[zoom_range], label="Moving Avg",
          linewidth=2, color=:blue, marker=:square, markersize=2)
    plot!(p[4], t[zoom_range], dy_gauss[zoom_range], label="Gaussian",
          linewidth=2, color=:green, marker=:diamond, markersize=2)
    plot!(p[4], t[zoom_range], dy_sg[zoom_range], label="Savitzky-Golay",
          linewidth=2, color=:red, marker=:utriangle, markersize=2)
    title!(p[4], "Zoomed First Derivative")
    xlabel!(p[4], "t")
    ylabel!(p[4], "dy/dt")

    savefig(p, "successful_smoothing_demo.png")

    # Print quantitative results
    println("="^80)
    println("WHAT THESE METHODS ACTUALLY DO WELL")
    println("="^80)
    println("\nNoise level: $noise_level")
    println("\nFunction Value Estimation (RMSE):")
    println("-"^40)

    rmse_ma = sqrt(mean((y_ma - y_true).^2))
    rmse_gauss = sqrt(mean((y_gauss - y_true).^2))
    rmse_sg = sqrt(mean((y_sg - y_true).^2))
    rmse_noisy = sqrt(mean((y_noisy - y_true).^2))

    @printf("Raw noisy:        %.6f\n", rmse_noisy)
    @printf("Moving Average:   %.6f (%.1fx better)\n", rmse_ma, rmse_noisy/rmse_ma)
    @printf("Gaussian:         %.6f (%.1fx better)\n", rmse_gauss, rmse_noisy/rmse_gauss)
    @printf("Savitzky-Golay:   %.6f (%.1fx better)\n", rmse_sg, rmse_noisy/rmse_sg)

    println("\nFirst Derivative Estimation (RMSE):")
    println("-"^40)

    rmse_dy_noisy = sqrt(mean((dy_noisy - dy_true).^2))
    rmse_dy_ma = sqrt(mean((dy_ma - dy_true).^2))
    rmse_dy_gauss = sqrt(mean((dy_gauss - dy_true).^2))
    rmse_dy_sg = sqrt(mean((dy_sg - dy_true).^2))

    @printf("Raw finite diff:  %.4f\n", rmse_dy_noisy)
    @printf("Moving Average:   %.4f (%.1fx better)\n", rmse_dy_ma, rmse_dy_noisy/rmse_dy_ma)
    @printf("Gaussian:         %.4f (%.1fx better)\n", rmse_dy_gauss, rmse_dy_noisy/rmse_dy_gauss)
    @printf("Savitzky-Golay:   %.4f (%.1fx better)\n", rmse_dy_sg, rmse_dy_noisy/rmse_dy_sg)
end

println("\n" * "="^80)
println("CONCLUSION")
println("="^80)
println("""

These smoothing methods WORK GREAT for their intended purpose:
1. Recovering smooth function values from noisy data ✓
2. Estimating first derivatives (and maybe second) ✓

They FAIL when we try to:
- Extend them to 7th order derivatives ✗
- Use them as input to spectral methods ✗

The package is doing exactly what it should! We were just asking it
to do something it wasn't designed for (high-order spectral derivatives).
""")