# Integration Plan: Derivative Estimation Study

## Current Status Summary

### What We've Learned from PyNumDiff Testing:

#### ✅ **GOOD Methods to Integrate** (RMSE < 0.1 on test function)
1. **butterdiff** - RMSE: 0.029 - Excellent overall performer
2. **TV regularized (tvrdiff)** - RMSE: 0.038 - Best for mixed signals
3. **TV velocity** - RMSE: 0.038 - Excellent boundary handling
4. **Polynomial fitting (polydiff)** - RMSE: 0.045 - Designed for polynomial signals
5. **Second-order finite diff** - RMSE: 0.074 - Simple and effective
6. **Spline diff** - RMSE: 0.092 - Good smooth approximation

#### ⚠️ **OK Methods** (RMSE 0.1-0.5) - Include with caveats
- Mediandiff - Works but has boundary issues
- Fourth-order FD - Moderate performance
- First-order FD - Baseline method

#### ❌ **SKIP These Methods** (fundamentally inappropriate)
- **RBF** (RMSE: 719!) - Catastrophic failure, conditioning issues
- **Kalman filters** (RMSE: 4.2) - Wrong model for polynomial growth
- **Spectral** (RMSE: 1.7) - Needs periodic signals
- **Window-based smoothers** - Severe boundary corruption

## Integration Tasks

### 1. Update PyNumDiff Wrapper (`methods/python/pynumdiff_wrapper/pynumdiff_methods.py`)

```python
# Add these successful methods if not already present:
- butterdiff (already there ✓)
- tvrdiff (add if missing)
- tv_velocity (already there ✓)
- polydiff (add)
- second_order_fd (add)
- splinediff (already there ✓)

# Mark these as "not recommended" or remove:
- rbfdiff
- kalman methods (for non-tracking problems)
- spectral (for non-periodic signals)
```

### 2. Create Method Selection Guide

```python
def select_pynumdiff_method(signal_properties):
    """
    Intelligent method selection based on signal characteristics
    """
    if signal_properties['has_polynomial_trend'] and signal_properties['has_oscillations']:
        return 'tv_velocity'  # Best for mixed signals
    elif signal_properties['is_smooth']:
        return 'butterdiff'  # Best overall for smooth signals
    elif signal_properties['has_polynomial_trend']:
        return 'polydiff'  # Designed for polynomials
    elif signal_properties['is_noisy']:
        return 'tv_regularized'  # Good noise handling
    else:
        return 'second_order_fd'  # Reliable fallback
```

### 3. Update Comprehensive Study Configuration

In `config.toml` or equivalent, add:

```toml
[methods.pynumdiff]
# High-performance methods
butterdiff = { enabled = true, category = "excellent" }
tv_velocity = { enabled = true, category = "excellent" }
tv_regularized = { enabled = true, category = "excellent" }
polydiff = { enabled = true, category = "good" }
splinediff = { enabled = true, category = "good" }

# Baseline methods
second_order_fd = { enabled = true, category = "baseline" }
fourth_order_fd = { enabled = false, category = "baseline" }

# Skip these (or mark for specific use cases only)
rbfdiff = { enabled = false, reason = "conditioning_issues" }
kalman_cv = { enabled = false, reason = "model_mismatch_for_polynomials" }
spectral = { enabled = false, reason = "requires_periodic_signals" }
```

### 4. Create Comprehensive Test Suite

```julia
# In src/comprehensive_study.jl, add test functions:

test_functions = [
    # Polynomial
    (t -> t.^2, "quadratic"),
    (t -> t.^(5/2), "power_2.5"),

    # Oscillatory
    (t -> sin.(2π*t), "sine"),
    (t -> cos.(3*t) + sin.(5*t), "multi_freq"),

    # Mixed (our test case)
    (t -> t.^(5/2) + sin.(2*t), "polynomial_oscillatory"),

    # Step/discontinuous
    (t -> sign.(t - 0.5), "step"),

    # Exponential
    (t -> exp.(t), "exponential"),
]
```

### 5. Performance Comparison Framework

Create a unified comparison that includes:
- Julia methods (finite diff, spectral, splines, GP, etc.)
- Python methods (from PyNumDiff and custom implementations)
- Hybrid approaches

```julia
function compare_all_methods(test_function, noise_level)
    results = Dict()

    # Julia methods
    results["julia_fd2"] = test_finite_diff(test_function, 2)
    results["julia_spectral"] = test_spectral(test_function)
    results["julia_spline"] = test_spline(test_function)

    # Python/PyNumDiff methods
    results["pynumdiff_butter"] = test_pynumdiff("butterdiff", test_function)
    results["pynumdiff_tv"] = test_pynumdiff("tv_velocity", test_function)

    return results
end
```

### 6. Generate Final Report

Structure:
```
1. Executive Summary
   - Best methods overall: TV regularization, Butterworth
   - Best for specific cases

2. Method Categories
   - Model-free: Finite differences, TV
   - Model-based: Kalman (for tracking only), Polynomial
   - Frequency-domain: Spectral (for periodic only)

3. Performance Tables
   - By noise level
   - By derivative order
   - By signal type

4. Recommendations
   - Default: Use TV regularization or Butterworth
   - High accuracy needed: Combine multiple methods
   - Real-time: Use simple finite differences
   - Known structure: Use appropriate model
```

## Implementation Priority

### Phase 1: Clean Integration (Immediate)
1. ✅ Update PyNumDiff wrapper with successful methods
2. ✅ Remove/disable failing methods
3. ✅ Add parameter recommendations

### Phase 2: Comprehensive Testing (This Week)
1. Run all good methods on standard test suite
2. Generate performance matrices
3. Identify best method for each signal class

### Phase 3: Documentation (Next Week)
1. Create method selection flowchart
2. Write usage guidelines
3. Generate publication-ready figures

### Phase 4: Advanced Features (Future)
1. Ensemble methods (combine multiple approaches)
2. Adaptive method selection
3. Uncertainty quantification

## Key Insights to Document

1. **Total Variation methods are the winners** for mixed polynomial+oscillatory signals
2. **Kalman filters fail on polynomials** - they need approximately constant velocity/acceleration
3. **RBF is catastrophically bad** for polynomial growth (conditioning issues)
4. **Simple methods often win** - Second-order FD is surprisingly robust
5. **Boundary handling matters** - TV methods excel here

## Code Organization

```
methods/
├── julia/
│   ├── finite_diff/
│   ├── spectral/
│   └── ...
├── python/
│   ├── pynumdiff_wrapper/
│   │   ├── pynumdiff_methods.py (UPDATE THIS)
│   │   └── method_selector.py (CREATE THIS)
│   └── ...
└── comparison/
    ├── unified_interface.jl (CREATE THIS)
    └── performance_metrics.jl (CREATE THIS)
```

## Next Steps

1. **Update PyNumDiff wrapper** to include only successful methods
2. **Create unified test harness** for all methods
3. **Run comprehensive comparison** on multiple test functions
4. **Generate final report** with recommendations

This will give you a production-ready derivative estimation toolkit with clear guidance on which methods to use when!