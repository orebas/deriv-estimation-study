"""
Investigate the boundary problems in smoothing methods
"""

using Plots
using Statistics
using Random
using Printf

gr()

# Generate test signal
n = 201
t = collect(range(0, 1, length=n))
dt = t[2] - t[1]

# True signal and derivatives
y_true = sin.(2π .* t)
dy_true = 2π .* cos.(2π .* t)
d2y_true = -(2π)^2 .* sin.(2π .* t)

# Add noise
Random.seed!(42)
noise_level = 1e-3
y_noisy = y_true .+ noise_level .* randn(n)

# Moving average with boundary issues
function moving_average_bad(y, window=11)
    n = length(y)
    y_smooth = similar(y)
    half_win = window ÷ 2
    for i in 1:n
        i_start = max(1, i - half_win)
        i_end = min(n, i + half_win)
        y_smooth[i] = mean(y[i_start:i_end])  # Problem: asymmetric near boundaries!
    end
    return y_smooth
end

# Better: Pad the signal to handle boundaries
function moving_average_padded(y, window=11)
    n = length(y)
    half_win = window ÷ 2

    # Pad with reflected values
    y_padded = vcat(
        reverse(y[2:half_win+1]),      # Reflect left
        y,
        reverse(y[end-half_win:end-1])  # Reflect right
    )

    y_smooth = similar(y)
    for i in 1:n
        # Now we can always use full window
        padded_idx = i + half_win
        y_smooth[i] = mean(y_padded[padded_idx-half_win:padded_idx+half_win])
    end
    return y_smooth
end

# Savitzky-Golay with boundary handling
function savitzky_golay_bad(y, window=11, order=3)
    n = length(y)
    y_smooth = similar(y)
    half_win = window ÷ 2

    for i in 1:n
        i_start = max(1, i - half_win)
        i_end = min(n, i + half_win)
        local_t = collect(i_start:i_end) .- i
        local_y = y[i_start:i_end]

        # Problem: Different window sizes near boundaries!
        A = zeros(length(local_t), min(order+1, length(local_t)))
        for j in 0:min(order, length(local_t)-1)
            A[:, j+1] = local_t.^j
        end

        coeffs = A \ local_y
        y_smooth[i] = coeffs[1]
    end
    return y_smooth
end

function savitzky_golay_extrapolated(y, window=11, order=3)
    n = length(y)
    y_smooth = similar(y)
    half_win = window ÷ 2

    # Fit polynomial to early/late portions for extrapolation
    fit_points = 20

    # Left extrapolation
    t_left = collect(1:fit_points)
    A_left = zeros(fit_points, order+1)
    for j in 0:order
        A_left[:, j+1] = t_left.^j
    end
    coeffs_left = A_left \ y[1:fit_points]

    # Right extrapolation
    t_right = collect(n-fit_points+1:n) .- (n-fit_points)
    A_right = zeros(fit_points, order+1)
    for j in 0:order
        A_right[:, j+1] = t_right.^j
    end
    coeffs_right = A_right \ y[n-fit_points+1:n]

    # Apply S-G with extrapolation
    for i in 1:n
        i_start = i - half_win
        i_end = i + half_win

        local_t = collect(i_start:i_end) .- i
        local_y = zeros(length(local_t))

        for (j, idx) in enumerate(i_start:i_end)
            if idx < 1
                # Use left extrapolation
                t_ext = idx
                local_y[j] = sum(coeffs_left[k+1] * t_ext^k for k in 0:order)
            elseif idx > n
                # Use right extrapolation
                t_ext = idx - (n-fit_points)
                local_y[j] = sum(coeffs_right[k+1] * t_ext^k for k in 0:order)
            else
                local_y[j] = y[idx]
            end
        end

        # Now fit polynomial to this data
        A = zeros(length(local_t), order+1)
        for j in 0:order
            A[:, j+1] = local_t.^j
        end

        coeffs = A \ local_y
        y_smooth[i] = coeffs[1]
    end
    return y_smooth
end

# Compute derivatives
function compute_derivative(y_smooth, dt)
    n = length(y_smooth)
    dy = zeros(n)

    # Use 5-point stencil where possible
    for i in 3:n-2
        dy[i] = (-y_smooth[i+2] + 8*y_smooth[i+1] - 8*y_smooth[i-1] + y_smooth[i-2]) / (12*dt)
    end

    # 3-point at near-boundaries
    dy[2] = (-y_smooth[4] + 4*y_smooth[3] - 3*y_smooth[2]) / (2*dt)
    dy[n-1] = (3*y_smooth[n-1] - 4*y_smooth[n-2] + y_smooth[n-3]) / (2*dt)

    # 2-point at boundaries
    dy[1] = (y_smooth[2] - y_smooth[1]) / dt
    dy[n] = (y_smooth[n] - y_smooth[n-1]) / dt

    return dy
end

# Apply different approaches
y_ma_bad = moving_average_bad(y_noisy)
y_ma_good = moving_average_padded(y_noisy)
y_sg_bad = savitzky_golay_bad(y_noisy)
y_sg_good = savitzky_golay_extrapolated(y_noisy)

# Compute derivatives
dy_ma_bad = compute_derivative(y_ma_bad, dt)
dy_ma_good = compute_derivative(y_ma_good, dt)
dy_sg_bad = compute_derivative(y_sg_bad, dt)
dy_sg_good = compute_derivative(y_sg_good, dt)

# Create diagnostic plots
p = plot(layout=(2,3), size=(1400, 600))

# Plot 1: Function values near left boundary
boundary_left = 1:30
plot!(p[1], t[boundary_left], y_true[boundary_left], label="True",
      linewidth=3, color=:black, title="Left Boundary - Function")
plot!(p[1], t[boundary_left], y_ma_bad[boundary_left], label="MA (bad)",
      linewidth=2, color=:red, linestyle=:dash)
plot!(p[1], t[boundary_left], y_ma_good[boundary_left], label="MA (padded)",
      linewidth=2, color=:green)
xlabel!(p[1], "t")
ylabel!(p[1], "y")

# Plot 2: Function values near right boundary
boundary_right = 172:201
plot!(p[2], t[boundary_right], y_true[boundary_right], label="True",
      linewidth=3, color=:black, title="Right Boundary - Function")
plot!(p[2], t[boundary_right], y_sg_bad[boundary_right], label="SG (bad)",
      linewidth=2, color=:red, linestyle=:dash)
plot!(p[2], t[boundary_right], y_sg_good[boundary_right], label="SG (extrap)",
      linewidth=2, color=:green)
xlabel!(p[2], "t")
ylabel!(p[2], "y")

# Plot 3: Full derivative comparison
plot!(p[3], t, dy_true, label="True", linewidth=3, color=:black,
      title="Full Derivative View")
plot!(p[3], t, dy_ma_bad, label="MA (bad)", linewidth=2, color=:red, alpha=0.7)
plot!(p[3], t, dy_sg_bad, label="SG (bad)", linewidth=2, color=:orange, alpha=0.7)
xlabel!(p[3], "t")
ylabel!(p[3], "dy/dt")

# Plot 4: Derivative near left boundary
plot!(p[4], t[boundary_left], dy_true[boundary_left], label="True",
      linewidth=3, color=:black, title="Left Boundary - Derivative")
plot!(p[4], t[boundary_left], dy_ma_bad[boundary_left], label="MA (bad)",
      linewidth=2, color=:red, linestyle=:dash, marker=:x)
plot!(p[4], t[boundary_left], dy_ma_good[boundary_left], label="MA (padded)",
      linewidth=2, color=:green, marker=:circle)
xlabel!(p[4], "t")
ylabel!(p[4], "dy/dt")

# Plot 5: Derivative near right boundary
plot!(p[5], t[boundary_right], dy_true[boundary_right], label="True",
      linewidth=3, color=:black, title="Right Boundary - Derivative")
plot!(p[5], t[boundary_right], dy_sg_bad[boundary_right], label="SG (bad)",
      linewidth=2, color=:red, linestyle=:dash, marker=:x)
plot!(p[5], t[boundary_right], dy_sg_good[boundary_right], label="SG (extrap)",
      linewidth=2, color=:green, marker=:circle)
xlabel!(p[5], "t")
ylabel!(p[5], "dy/dt")

# Plot 6: Error map
error_ma = abs.(dy_ma_bad - dy_true)
error_sg = abs.(dy_sg_bad - dy_true)
plot!(p[6], t, error_ma, label="MA error", linewidth=2, color=:red,
      title="Derivative Error vs Position", yaxis=:log)
plot!(p[6], t, error_sg, label="SG error", linewidth=2, color=:orange)
xlabel!(p[6], "t")
ylabel!(p[6], "|error|")

# Add vertical lines showing "affected zone"
vline!(p[6], [0.05, 0.95], color=:gray, linestyle=:dash, label="5% boundaries")
vline!(p[6], [0.025, 0.975], color=:gray, linestyle=:dot, label="2.5% boundaries")

savefig(p, "boundary_problems_analysis.png")

# Quantitative analysis
println("="^80)
println("BOUNDARY EFFECTS ANALYSIS")
println("="^80)

# Define zones
inner_90 = 11:191  # Middle 90%
inner_80 = 21:181  # Middle 80%
boundary_10 = vcat(1:10, 192:201)  # 10% at each end

println("\nDerivative RMSE by region:")
println("-"^50)

for (name, dy_est) in [("Moving Avg (bad)", dy_ma_bad),
                        ("Moving Avg (padded)", dy_ma_good),
                        ("Savitzky-Golay (bad)", dy_sg_bad),
                        ("Savitzky-Golay (extrap)", dy_sg_good)]

    rmse_full = sqrt(mean((dy_est - dy_true).^2))
    rmse_inner80 = sqrt(mean((dy_est[inner_80] - dy_true[inner_80]).^2))
    rmse_boundary = sqrt(mean((dy_est[boundary_10] - dy_true[boundary_10]).^2))

    @printf("%-25s | Full: %.4f | Inner 80%%: %.4f | Boundaries: %.4f\n",
            name, rmse_full, rmse_inner80, rmse_boundary)
end

println("\nHow far do boundary effects extend?")
println("-"^50)

# Find where error exceeds 2x the minimum error
for (name, dy_est) in [("Moving Avg", dy_ma_bad), ("Savitzky-Golay", dy_sg_bad)]
    errors = abs.(dy_est - dy_true)
    min_error = minimum(errors[inner_80])
    threshold = 2 * min_error

    # Find extent from left
    left_extent = findfirst(e -> e < threshold, errors)
    # Find extent from right
    right_extent = findlast(e -> e < threshold, errors)

    if !isnothing(left_extent) && !isnothing(right_extent)
        left_pct = 100 * left_extent / n
        right_pct = 100 * (n - right_extent) / n
        @printf("%s: Effects extend %.1f%% from left, %.1f%% from right\n",
                name, left_pct, right_pct)
    end
end

println("\nConclusion: Boundary effects can extend 5-15% into the domain!")
println("This is why many methods trim boundaries or use special handling.")