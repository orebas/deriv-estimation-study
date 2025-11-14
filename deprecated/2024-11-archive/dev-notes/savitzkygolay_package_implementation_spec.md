# SavitzkyGolay.jl Package Implementation Specification
## Expert Consultation Summary: GPT-5 + Gemini-2.5-Pro

Date: 2025-10-30
Consultants: GPT-5, Gemini-2.5-Pro
Context: Derivative estimation benchmark study (N=251, smooth ODEs, noise 1e-8 to 2e-2)

---

## Executive Summary

**Core Insight**: Optimal window size should be specified in **physical units** `h` (domain scale), not discrete samples `w`. The physical window `h ≈ 0.15-0.18` represents the timescale that best separates signal dynamics from noise and remains constant across different N values.

**Recommended Strategy**: Implement THREE methods for comprehensive benchmarking:
1. **SG-Package-Fixed**: Fixed physical window (robust baseline)
2. **SG-Package-Hybrid**: Gentle adaptive adjustment (best of both worlds)
3. **SG-Package-Adaptive**: Full adaptive (theoretical comparison)

---

## Method 1: SG-Package-Fixed

**Purpose**: Robust, physically-grounded baseline method

### Parameters

#### Physical Window (by derivative order and polynomial order)
Base physical window `h_base(p, r)` from GPT-5 calibration:

**For p = 7:**
- r = 0-2: h = 0.16
- r = 3-4: h = 0.18
- r = 5: h = 0.20
- r = 6-7: h = 0.22

**For p = 9:**
- r = 0-2: h = 0.15
- r = 3-4: h = 0.17
- r = 5: h = 0.19
- r = 6-7: h = 0.21

**For p = 11 (only for very low noise σ < 5e-6):**
- r = 0-2: h = 0.14
- r = 3-4: h = 0.16
- r = 5: h = 0.18
- r = 6-7: h = 0.20

#### Polynomial Order Selection
```julia
function select_polyorder(deriv_order::Int, noise_estimate::Float64, signal_scale::Float64)
    σ_relative = noise_estimate / signal_scale

    # Base tier: p = clamp(r + 2, 5, 9)
    p = clamp(deriv_order + 2, 5, 9)

    # Special case: very low noise + high derivative → allow p=11
    if σ_relative ≤ 5e-6 && deriv_order ≥ 5
        p = min(deriv_order + 4, 11)
    end

    # Special case: high noise → reduce to minimum
    if σ_relative ≥ 1e-2 && deriv_order ≥ 3
        p = max(deriv_order + 1, 5)
    end

    return p
end
```

**Simplified version (recommended for initial implementation):**
```julia
# Simple tiered approach
p = clamp(deriv_order + 2, 5, 9)
```

#### Window Size Computation
```julia
function compute_window_size(h_physical::Float64, dx::Float64, n::Int, polyorder::Int)
    # Convert physical window to discrete samples
    w_ideal = 2 * floor(Int, h_physical / dx) + 1

    # Apply constraints
    w_min = polyorder + 3  # Numerical stability (not just p+2)
    w_max = n ÷ 3          # Cap at 1/3 of data size

    w = clamp(w_ideal, w_min, min(w_max, n))

    # Ensure odd
    w = isodd(w) ? w : w - 1

    return w
end
```

### Boundary Handling

**GPT-5 recommendation (derivative-order dependent):**
```julia
function compute_boundary_discard(window_size::Int, deriv_order::Int)
    hw = (window_size - 1) ÷ 2

    if deriv_order ≤ 2
        return ceil(Int, 0.5 * hw)  # Discard ~50% of half-window
    elseif deriv_order ≤ 4
        return ceil(Int, 0.75 * hw) # Discard ~75% of half-window
    else  # deriv_order ≥ 5
        return hw                    # Discard full half-window
    end
end
```

**Example for w=51 (hw=25):**
- r ≤ 2: discard 13 points each end
- r = 3-4: discard 19 points each end
- r ≥ 5: discard 25 points each end

**Gemini recommendation (simpler, uniform):**
```julia
# Discard full half-window for all orders
boundary_discard = (window_size - 1) ÷ 2
```

**Recommendation**: Start with GPT-5's approach; it's more rigorous for high-order derivatives.

### Implementation Pseudocode
```julia
function evaluate_sg_package_fixed(x, y, x_eval, orders; params=Dict())
    n = length(x)
    dx = mean(diff(x))
    signal_scale = maximum(abs.(y))

    results = Dict()
    for deriv_order in orders
        # Select polynomial order
        p = clamp(deriv_order + 2, 5, 9)

        # Get physical window from calibration table
        h = get_physical_window(p, deriv_order)  # Use table above

        # Compute discrete window size
        w = compute_window_size(h, dx, n, p)

        # Call package
        result = savitzky_golay(y, w, p; deriv=deriv_order, rate=1/dx)

        # Discard boundaries
        m = compute_boundary_discard(w, deriv_order)
        valid_indices = (m+1):(n-m)

        # Interpolate to x_eval (only from valid interior points)
        deriv_eval = linear_interpolate(x[valid_indices], result.y[valid_indices], x_eval)

        results[deriv_order] = deriv_eval
    end

    return results
end
```

---

## Method 2: SG-Package-Hybrid

**Purpose**: Gentle adaptive adjustment around fixed baseline (GPT-5 recommendation)

### Strategy
Blend fixed physical window with adaptive formula:
```julia
h = clamp((1-α)*h_base + α*h_adapt, h_min, h_max)
```
where `α = 0.3` (gentle blending factor)

### Rationale
- MISE formula has very shallow exponent (1/17 for p=7), so full adaptivity barely moves window
- Noise/roughness estimation has uncertainty; gentle blending provides stability
- Allows response to extreme cases (very high/low noise) without instability

### Adaptive Component
```julia
function compute_adaptive_window(σ_hat::Float64, ρ_hat::Float64, p::Int, r::Int, dx::Float64)
    # Get calibration constant (same as fixed baseline)
    c = get_physical_window(p, r)

    # Compute adaptive physical window
    # Add safeguards to prevent numerical issues
    σ_safe = max(σ_hat, 1e-24)
    ρ_safe = max(ρ_hat, 1e-24)

    exponent = 1.0 / (2*p + 3)
    h_adapt = c * (σ_safe^2 / ρ_safe^2)^exponent

    return h_adapt
end
```

### Noise Estimation
```julia
# Use wavelet MAD (gold standard)
σ_hat = HyperparameterSelection.estimate_noise_wavelet(y)
```

### Roughness Estimation
```julia
# 4th-order differences (Rice 1984)
function estimate_roughness_4th_order(y::Vector{Float64}, dx::Float64)
    d4 = y
    for _ in 1:4
        d4 = diff(d4)
    end
    ρ_hat = sqrt(mean(d4.^2)) / (dx^4 + 1e-24)
    return ρ_hat
end
```

### Hybrid Blending
```julia
function compute_hybrid_window(h_base::Float64, h_adapt::Float64, p::Int, r::Int)
    α = 0.3  # Gentle blending (30% adaptive, 70% fixed)

    # Physical window bounds
    h_min = 0.10
    h_max = 0.28

    h = (1 - α) * h_base + α * h_adapt
    h = clamp(h, h_min, h_max)

    return h
end
```

### Full Implementation
```julia
function evaluate_sg_package_hybrid(x, y, x_eval, orders; params=Dict())
    n = length(x)
    dx = mean(diff(x))
    signal_scale = maximum(abs.(y))

    # Estimate noise and roughness once (same for all orders)
    σ_hat = HyperparameterSelection.estimate_noise_wavelet(y)
    ρ_hat = estimate_roughness_4th_order(y, dx)

    results = Dict()
    for deriv_order in orders
        # Select polynomial order
        p = clamp(deriv_order + 2, 5, 9)

        # Get baseline physical window
        h_base = get_physical_window(p, deriv_order)

        # Compute adaptive suggestion
        h_adapt = compute_adaptive_window(σ_hat, ρ_hat, p, deriv_order, dx)

        # Blend
        h = compute_hybrid_window(h_base, h_adapt, p, deriv_order)

        # Convert to discrete window
        w = compute_window_size(h, dx, n, p)

        # Call package
        result = savitzky_golay(y, w, p; deriv=deriv_order, rate=1/dx)

        # Discard boundaries
        m = compute_boundary_discard(w, deriv_order)
        valid_indices = (m+1):(n-m)

        # Interpolate to x_eval
        deriv_eval = linear_interpolate(x[valid_indices], result.y[valid_indices], x_eval)

        results[deriv_order] = deriv_eval
    end

    return results
end
```

---

## Method 3: SG-Package-Adaptive

**Purpose**: Pure theoretical approach for comparison and failure analysis

### Strategy
Use MISE formula directly with minimal constraints.

### Implementation
```julia
function evaluate_sg_package_adaptive(x, y, x_eval, orders; params=Dict())
    n = length(x)
    dx = mean(diff(x))

    # Estimate noise and roughness
    σ_hat = HyperparameterSelection.estimate_noise_wavelet(y)
    ρ_hat = estimate_roughness_4th_order(y, dx)

    results = Dict()
    for deriv_order in orders
        # Select polynomial order
        p = clamp(deriv_order + 2, 5, 9)

        # Compute adaptive window (pure formula, no baseline)
        c = get_physical_window(p, deriv_order)  # Use as scaling constant
        σ_safe = max(σ_hat, 1e-24)
        ρ_safe = max(ρ_hat, 1e-24)

        exponent = 1.0 / (2*p + 3)
        h_adapt = c * (σ_safe^2 / ρ_safe^2)^exponent

        # Apply hard bounds (prevent pathological cases)
        h = clamp(h_adapt, 0.05, 0.35)

        # Convert to discrete window
        w = compute_window_size(h, dx, n, p)

        # Call package
        result = savitzky_golay(y, w, p; deriv=deriv_order, rate=1/dx)

        # Discard boundaries
        m = compute_boundary_discard(w, deriv_order)
        valid_indices = (m+1):(n-m)

        # Interpolate to x_eval
        deriv_eval = linear_interpolate(x[valid_indices], result.y[valid_indices], x_eval)

        results[deriv_order] = deriv_eval
    end

    return results
end
```

---

## Edge Case Handling

### Very High Noise (σ = 2e-2, SNR ≈ 1)

**Fixed Method:**
```julia
if σ_relative ≥ 1e-2
    h *= 1.3  # Increase window by 30%
    h = min(h, 0.28)  # Cap at maximum
    p = max(deriv_order + 1, 5)  # Reduce to minimum polyorder
end
```

**Expected behavior**: Trade spatial resolution for variance reduction. High-order derivatives (r≥5) will still be very noisy.

### Very Low Noise (σ = 1e-8)

**Fixed Method:**
```julia
if σ_relative ≤ 5e-6
    h *= 0.7  # Decrease window by 30% to reduce bias
    h = max(h, 0.10)  # Maintain minimum
    if deriv_order ≥ 5
        p = min(deriv_order + 4, 11)  # Allow higher polyorder
    end
end
```

**Adaptive Method:**
Must add lower bound to prevent window collapse:
```julia
w_min = max(2*deriv_order + 5, 11)  # More aggressive minimum
```

### Higher Derivative Orders (r = 5, 6, 7)

**Key challenges:**
- Noise amplification: Variance ∝ σ²/h^(2r+1)
- Numerical scaling: rate^r * factorial(r) becomes very large
- Boundary effects: More sensitive to edge artifacts

**Recommendations:**
1. Use larger physical windows (h ≥ 0.20)
2. Ensure p ≥ r + 2 at minimum
3. Discard full half-window boundaries
4. Scale signal to O(1) before processing to avoid roundoff
5. Report these estimates with appropriate uncertainty warnings

---

## Implementation Checklist

### Phase 1: Core Implementation
- [ ] Create `methods/julia/filtering/savitzky_golay_package.jl`
- [ ] Implement helper functions:
  - [ ] `get_physical_window(p, r)` - calibration table lookup
  - [ ] `select_polyorder(r, σ_relative)` - polynomial order selection
  - [ ] `compute_window_size(h, dx, n, p)` - h → w conversion
  - [ ] `compute_boundary_discard(w, r)` - boundary points to discard
  - [ ] `estimate_roughness_4th_order(y, dx)` - roughness estimation
- [ ] Implement three main methods:
  - [ ] `evaluate_sg_package_fixed(x, y, x_eval, orders)`
  - [ ] `evaluate_sg_package_hybrid(x, y, x_eval, orders)`
  - [ ] `evaluate_sg_package_adaptive(x, y, x_eval, orders)`

### Phase 2: Integration
- [ ] Register methods in `src/julia_methods_integrated.jl`
- [ ] Update `python/generate_additional_figures.py` to include new methods
- [ ] Update `python/generate_paper_tables.py` to include new methods
- [ ] Update `config.toml` if needed for method-specific parameters

### Phase 3: Testing
- [ ] Unit tests for helper functions
- [ ] Comparison test: package vs our implementation (same parameters → same results)
- [ ] Edge case tests: very high noise, very low noise, high derivatives
- [ ] Boundary handling verification

### Phase 4: Benchmarking
- [ ] Run comprehensive study with all three new methods
- [ ] Compare performance across noise levels and derivative orders
- [ ] Analyze failure modes of adaptive method
- [ ] Generate comparative figures

---

## Expected Results

### Method Performance Hierarchy (anticipated)

**At N=251 with moderate noise (σ = 1e-4 to 1e-3):**
1. **Hybrid**: Best overall performance (stable + slight adaptation)
2. **Fixed**: Very close to Hybrid, simpler to explain
3. **Adaptive**: Similar to Hybrid for N=251, but may fail at extremes

**At extreme noise (σ = 2e-2):**
1. **Hybrid/Fixed**: Will gracefully degrade with controlled smoothing
2. **Adaptive**: May struggle if roughness estimate contaminated

**At very low noise (σ = 1e-8):**
1. **Fixed**: Consistent performance
2. **Hybrid**: May over-adapt (reduce window too much)
3. **Adaptive**: Risk of window collapse without proper guards

**At high derivatives (r = 6-7):**
- All methods will show high errors due to fundamental noise amplification
- Fixed method will be most predictable

---

## Validation Strategy

### 1. Sanity Checks
- Chosen w should vary slowly with noise (factor ~1.1-1.3 across range)
- Physical window h should remain in reasonable bounds (0.10-0.28)
- Polynomial order should satisfy p ≥ r + 2

### 2. Diagnostic Plots
- Window size w vs noise level σ (for each method)
- nRMSE vs derivative order (compare three methods)
- Boundary effect visualization (show discarded regions)

### 3. Comparative Analysis
- Fixed vs Hybrid: Should be very close, Hybrid slightly better at extremes
- Hybrid vs Adaptive: Hybrid should be more stable
- Package vs Our Implementation: Should match within numerical tolerance

---

## Publication Recommendations

### Main Result
Present **Fixed** method as the primary recommendation:
- Simple, robust, physically interpretable
- Based on signal timescale, not arbitrary N-dependent window
- Easy to replicate and understand

### Methodological Contribution
Present **Hybrid** method to demonstrate:
- How theoretical adaptive methods can be stabilized
- Minimal performance gain suggests fixed approach is sufficient
- Useful for readers interested in adaptive methods

### Comparative Analysis
Use **Adaptive** method to:
- Show limitations of pure theoretical approaches at finite N
- Demonstrate failure modes (very low noise, small N)
- Validate that physical window insight is more robust than pure formula

---

## Key Insights for Paper

1. **Physical vs Discrete Window**: The key conceptual advance is recognizing that optimal window size is a physical quantity h (in domain units), not a sample count w. This makes the method generalizable across different discretizations.

2. **Fixed is Competitive**: Despite its simplicity, a well-chosen fixed physical window performs comparably to or better than sophisticated adaptive methods for this problem class.

3. **Asymptotic Theory Limitations**: MISE-based formulas derived under asymptotic assumptions (large N) can be unreliable at practical N, especially with accurate noise estimation (paradoxically, better noise estimates can lead to worse window choices).

4. **Polynomial Order**: The choice of polynomial order p is more important than previously recognized. Using p = r + 2 (rather than fixed high p) provides better bias-variance balance.

5. **Boundary Handling**: Derivative-order dependent boundary discard (more aggressive for high r) is a simple but effective refinement.

---

## References for Paper

1. **Savitzky & Golay (1964)**: Original paper - cite for historical context
2. **Rice (1984)**: Bandwidth selection in kernel regression - theory behind window selection
3. **Donoho & Johnstone (1994)**: Wavelet noise estimation - justifies our estimator choice
4. **Schafer (2011)**: IEEE Signal Processing Magazine - modern perspective on S-G filters

## Expert Consultants

This specification synthesizes recommendations from:
- **GPT-5** (OpenAI): Provided detailed calibration constants, hybrid blending strategy, derivative-order dependent boundary handling
- **Gemini-2.5-Pro** (Google): Emphasized physical window concept, advocated for simplicity, provided comparative method framework
