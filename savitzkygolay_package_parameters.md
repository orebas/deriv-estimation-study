# SavitzkyGolay.jl Package Parameters

## Function Signature

```julia
savitzky_golay(y, window_size, order; deriv=0, rate=1.0, haswts=false)
savitzky_golay(y, wts, window_size, order; deriv=0, rate=1.0, haswts=true)
```

## Required Parameters

### 1. `y::AbstractVector`
- Input data vector
- Will be our noisy observations

### 2. `window_size::Int` (must be odd)
- **This is THE key hyperparameter**
- Same as our `window` parameter
- Constraints:
  - Must be odd
  - Must be > order
  - Should be < n (data length)

### 3. `order::Int`
- Polynomial degree for fitting
- Same as our `polyorder` parameter
- Must be >= deriv (derivative order)

## Keyword Parameters

### 4. `deriv::Int` (default: 0)
- Derivative order to compute
- 0 = smoothing only
- 1 = first derivative
- 2 = second derivative
- etc.

### 5. `rate::Float64` (default: 1.0)
- **CRITICAL:** Sampling rate = 1/dx (NOT dx!)
- From our testing: must use `rate = 1/dx` for correct scaling
- Examples:
  - dx = 0.01 → rate = 100
  - dx = 0.004 → rate = 250

### 6. `haswts::Bool` (default: false)
- Whether using weighted least squares
- We don't need this for now

### 7. `wts::AbstractVector` (optional, 2nd signature)
- Weights for each point in window
- Only if we want weighted S-G (future feature?)

## Design Decisions for Our Implementation

### Option 1: Fixed Methods (N-dependent)

Create fixed-parameter methods with sensible defaults based on N:

```julia
function evaluate_savitzky_golay_package_fixed(x, y, x_eval, orders; params=Dict())
    n = length(x)
    dx = mean(diff(x))

    # Sensible defaults based on data size
    default_window = if n <= 60
        9       # N≈51
    elseif n <= 150
        15      # N≈101
    else
        37      # N≈251
    end

    window_size = get(params, :sg_window, default_window)
    polyorder = get(params, :sg_polyorder, 7)  # Fixed poly=7 for simplicity

    # Ensure odd
    window_size = isodd(window_size) ? window_size : window_size + 1

    # Per-order evaluation
    for order in orders
        # Adjust polyorder if needed for higher orders
        p = order <= 3 ? polyorder : (order <= 5 ? 9 : 11)

        result = savitzky_golay(y, window_size, p;
                                deriv=order,
                                rate=1/dx)
        # ... interpolate to x_eval ...
    end
end
```

**Hyperparameters:**
- `:sg_window` (default: N-dependent, see above)
- `:sg_polyorder` (default: 7)

### Option 2: Adaptive Method (Noise-based)

Use the same adaptive algorithm but call the package:

```julia
function evaluate_savitzky_golay_package_adaptive(x, y, x_eval, orders; params=Dict())
    n = length(x)
    dx = mean(diff(x))

    # Noise estimation (wavelet MAD - gold standard)
    σ̂ = HyperparameterSelection.estimate_noise_wavelet(y)

    # Roughness estimation (4th-order differences)
    d4 = y
    for _ in 1:4; d4 = diff(d4); end
    ρ̂ = sqrt(mean(d4.^2)) / (dx^4 + 1e-24)

    # Calibration constants
    c_pr = Dict(
        (7, 0) => 1.0, (7, 1) => 1.1, (7, 2) => 1.2, (7, 3) => 1.3,
        (9, 4) => 1.4, (9, 5) => 1.5,
        (11, 6) => 1.6, (11, 7) => 1.7,
    )

    max_window = max(5, n ÷ 3)

    for order in orders
        # Polyorder tiers
        p = order <= 3 ? 7 : (order <= 5 ? 9 : 11)

        # Compute optimal window
        c = get(c_pr, (p, order), 1.0 + 0.1 * order)
        ratio = max(σ̂, 1e-24)^2 / max(ρ̂, 1e-24)^2
        h_star = c * (ratio^(1.0 / (2 * p + 3)))
        w_ideal = Int(2 * floor(h_star / dx) + 1)

        # Apply constraints
        w = max(w_ideal, p + 3)
        w = min(w, max_window)
        w = min(w, n)
        w = isodd(w) ? w : w - 1

        # Call package
        result = savitzky_golay(y, w, p;
                                deriv=order,
                                rate=1/dx)
        # ... interpolate to x_eval ...
    end
end
```

**Hyperparameters:**
- None (all automatic)
- Could add: `:max_window_fraction` (default: 1/3)
- Could add: `:noise_estimator` (choice: "wavelet" or "diff2")

### Option 3: Simple Wrapper (Minimal)

Just expose package parameters directly:

```julia
function evaluate_savitzky_golay_package(x, y, x_eval, orders; params=Dict())
    n = length(x)
    dx = mean(diff(x))

    # Direct parameter pass-through
    window_size = get(params, :window_size, 15)
    order = get(params, :order, 7)

    # Ensure constraints
    window_size = isodd(window_size) ? window_size : window_size + 1
    window_size = min(window_size, n)

    for deriv_order in orders
        result = savitzky_golay(y, window_size, order;
                                deriv=deriv_order,
                                rate=1/dx)
        # ... interpolate to x_eval ...
    end
end
```

**Hyperparameters:**
- `:window_size` (required or default=15)
- `:order` (required or default=7)

## Recommended Approach: All Three

Implement all three as separate methods:

1. **`SavitzkyGolay-Package-Fixed`** - N-dependent sensible defaults
2. **`SavitzkyGolay-Package-Adaptive`** - Noise-based window selection
3. (Optional) **`SavitzkyGolay-Package-Custom`** - Direct parameter control

This gives users flexibility:
- **Fixed**: "Just make it work" with good defaults
- **Adaptive**: "Optimize for my noise level"
- **Custom**: "I know exactly what I want"

## Critical Implementation Details

### 1. Rate Calculation
```julia
dx = mean(diff(x))  # Grid spacing
rate = 1 / dx       # Sampling rate (NOT dx!)
```

### 2. Boundary Handling
Package returns full-length vector (with boundaries computed via extrapolation).
We need to:
- Either use package's boundary values directly
- Or NaN them out and interpolate (consistent with our approach)

### 3. Interpolation to x_eval
Package gives results at training points `x`.
We need to interpolate to `x_eval` (may differ from `x`).

### 4. Error Handling
Check if package throws errors for:
- Window too large
- Window <= order
- Non-odd window

### 5. Polyorder per Derivative Order
Should we use different polynomial orders for different derivative orders?
- Orders 0-3: p=7
- Orders 4-5: p=9
- Orders 6-7: p=11

Or keep it simple with fixed p=7 (or let adaptive handle it)?

## Testing Strategy

Compare against our implementation:
1. Same parameters → same results (within numerical tolerance)
2. Different N → appropriate window scaling
3. Different noise levels → adaptive responds correctly
4. Boundary behavior → check edge cases

## Summary of Hyperparameters

| Method | Hyperparameters | Default Values |
|--------|----------------|----------------|
| **Package-Fixed** | `window_size`, `order` | N-dependent (9/15/37), 7 |
| **Package-Adaptive** | None (automatic) | Wavelet noise, formula-based |
| **Package-Custom** | `window_size`, `order` | User-specified |

All methods use:
- `rate = 1/dx` (computed automatically)
- `deriv = order` (loops over requested orders)
- `haswts = false` (no weighting for now)
