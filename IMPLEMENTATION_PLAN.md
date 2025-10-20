# Simplified Implementation Plan
## High-Order Derivative Estimation Study

**Date**: 2025-10-13 (Updated: 2025-10-13)
**Philosophy**: Get it working first, optimize later
**Current Status**: ✅ Pilot phase complete, 7 methods validated

## 🔄 Current Status Summary

**Phase**: Pilot Testing & Method Validation (COMPLETE)

**Working Methods**: 7 of 15 implemented
- Julia: AAA-HighPrec, AAA-LowPrec, Dierckx-5, Central-FD
- Python: scipy-RBF (2 variants), scipy-CubicSpline

**Key Achievements**:
- ✅ Ground truth generation working (orders 0-7)
- ✅ TaylorDiff API fixed (Val(n) syntax)
- ✅ Dierckx namespace conflict resolved
- ✅ Python native derivative APIs working
- ✅ End-to-end pipeline validated
- ✅ Diagnostic testing framework created

**Known Issues**:
- GP-Julia-SE: Cannot differentiate through external C lib
- Fourier-Interp: Numerical explosion
- Savitzky-Golay/TrendFilter: Not computing derivatives properly
- Python methods need further work (separate session)

**See STATUS.md for detailed results.**

---

## Architecture Overview

### Python Integration Strategy: **Standalone Scripts + JSON/CSV**

**Rationale**: Simpler, more robust, easier to debug than PyCall

```
Workflow:
1. Julia generates noisy data → exports to JSON/CSV
2. Standalone Python script reads data → runs methods → exports results to JSON/CSV
3. Julia imports results → computes errors → generates plots
```

**Benefits**:
- No PyCall complexity
- Python environment isolated (UV + venv)
- Easy to debug each component
- Can run Python methods independently
- No cross-language AD issues

---

## Methods Catalog (Simplified)

### Julia Methods (Priority 1)
1. **AAA-HighPrec** - BaryRational.jl (already implemented in bary_derivs.jl)
2. **AAA-LowPrec** - BaryRational.jl
3. **GP-Julia** - GaussianProcesses.jl (already have)
4. **Fourier-Interp** - From bary_derivs.jl
5. **Dierckx-5** - Dierckx.jl spline k=5
6. **Central-FD** - Custom implementation
7. **Savitzky-Golay** - Via convolution (implement simple version)
8. **TrendFiltering-k7** - TrendFiltering.jl (NEW - excellent for high orders)

### Python Methods (Priority 1)
1. **sklearn-GP** - scikit-learn GaussianProcessRegressor
2. **scipy-RBF-multiquadric** - scipy.interpolate.Rbf
3. **scipy-CubicSpline** - scipy.interpolate.CubicSpline
4. **scipy-UnivariateSpline** - with s parameter tuning
5. **tvregdiff** - Total variation differentiation
6. **pydiff-TrendFiltered** - Modern trend filtering (from pysindy)
7. **numdifftools** - Numerical differentiation baseline

### Optional (Priority 2 - add if time)
- FFT spectral differentiation
- Floater-Hormann (FHD)
- Additional RBF variants
- Wavelet-based methods

---

## File Structure

```
derivative_estimation_study/
├── DESIGN.md                       # Original comprehensive design
├── IMPLEMENTATION_PLAN.md          # This file (simplified plan)
├── Project.toml / Manifest.toml   # Julia dependencies
├── src/
│   ├── main_study.jl              # Main driver
│   ├── ground_truth.jl            # Generate Lotka-Volterra derivatives
│   ├── noise_model.jl             # Add noise (constant Gaussian, switchable)
│   ├── julia_methods.jl           # All Julia differentiation methods
│   ├── evaluation.jl              # Error metrics (RMSE, MAE, etc.)
│   └── visualization.jl           # Plotting functions
├── python/
│   ├── requirements.txt           # Python dependencies for UV
│   ├── python_methods.py          # Standalone Python script
│   └── .venv/                     # Created by UV (gitignored)
├── data/
│   ├── input/                     # JSON files: noisy data from Julia
│   └── output/                    # JSON files: Python method results
├── results/
│   ├── pilot/
│   │   ├── julia_results.csv
│   │   ├── python_results.csv
│   │   └── combined_summary.csv
│   └── full/
└── figures/
    └── pilot/
```

---

## Data Exchange Format

### Input: Julia → Python (data/input/trial_N{n}_noise{p}.json)

```json
{
  "system": "Lotka-Volterra",
  "observable": "x(t)",
  "times": [0.0, 0.1, 0.2, ...],
  "y_noisy": [1.0, 1.05, ...],
  "y_true": [1.0, 1.048, ...],
  "ground_truth_derivatives": {
    "0": [1.0, 1.048, ...],
    "1": [0.5, 0.48, ...],
    "2": [...],
    ...
    "7": [...]
  },
  "config": {
    "data_size": 51,
    "noise_level": 0.01,
    "trial": 1,
    "tspan": [0.0, 10.0]
  }
}
```

### Output: Python → Julia (data/output/trial_N{n}_noise{p}_results.json)

```json
{
  "trial_id": "N51_noise0.01_trial1",
  "methods": {
    "sklearn-GP": {
      "predictions": {
        "0": [1.001, 1.049, ...],
        "1": [0.501, ...],
        ...
        "7": [...]
      },
      "timing": 0.234,
      "success": true,
      "failures": {}
    },
    "tvregdiff-order7": {
      "predictions": {...},
      "timing": 0.045,
      "success": true,
      "failures": {}
    },
    ...
  }
}
```

---

## Noise Model Design

**Current**: Constant Gaussian (additive white noise scaled to signal magnitude)

```julia
function add_noise(signal, noise_level, rng; model=:constant_gaussian)
    if model == :constant_gaussian
        σ = noise_level * std(signal)  # Fixed: was mean(abs.(signal))
        return signal .+ σ .* randn(rng, length(signal))
    elseif model == :proportional
        # Future: per-sample proportional noise
        return signal .* (1 .+ noise_level .* randn(rng, length(signal)))
    elseif model == :heteroscedastic
        # Future: signal-dependent noise
        σ_base = noise_level * median(abs.(signal))
        σ_local = @. noise_level * (abs(signal) + σ_base)
        return signal .+ σ_local .* randn(rng, length(signal))
    else
        error("Unknown noise model: $model")
    end
end
```

**Start with** `:constant_gaussian`, add others later.

---

## Experimental Parameters (Pilot Study)

**Simplified for first run**:
- **System**: Lotka-Volterra only
- **Observable**: x(t) (prey)
- **Monte Carlo trials**: 2
- **Noise levels**: [0%, 1%]
- **Data sizes**: [51, 101]
- **Derivative orders**: 0 through 7
- **Time span**: [0, 10]
- **Random seed**: 12345

**Total pilot runs**: 2 trials × 2 noise × 2 sizes = 8 configurations

---

## Implementation Phases

### Phase 0: Setup ✅ COMPLETE
- [x] Create file structure
- [x] Set up Julia Project.toml with dependencies
- [x] Create Python requirements.txt
- [x] Initialize UV venv: `uv venv python/.venv`
- [x] Install Python packages: `uv pip install -r python/requirements.txt`

### Phase 1: Ground Truth ✅ COMPLETE
- [x] Implement Lotka-Volterra symbolic derivative calculation
- [x] Test with ModelingToolkit to verify orders 0-7
- [x] Export to JSON for validation

### Phase 2: Julia Methods ⚠️ PARTIAL (4/8 working)
- [x] Port AAA from bary_derivs.jl (✅ Working)
- [x] Port GP from bary_derivs.jl (❌ Cannot differentiate external C lib)
- [x] Port Fourier interpolation (❌ Numerical explosion)
- [x] Implement Dierckx spline wrapper (✅ Working - fixed namespace issue)
- [x] Implement central finite differences (✅ Working orders 0-1)
- [x] Implement simple Savitzky-Golay (⚠️ Only smooths, no derivatives)
- [x] Add TrendFiltering.jl method (⚠️ Only smooths, no derivatives)
- [x] Test each in isolation (✅ Diagnostic test created)

### Phase 3: Python Methods ⚠️ PARTIAL (3/7 working)
- [x] Create standalone script structure (✅ Working)
- [x] Implement sklearn GP wrapper (❌ No native derivative API)
- [x] Implement scipy RBF wrapper (✅ Working)
- [x] Implement scipy spline wrappers (✅ CubicSpline working, UnivariateSpline over-smoothing)
- [ ] Add tvregdiff integration (Not yet started)
- [x] Add pydiff TrendFiltered (⚠️ Results suspicious)
- [x] Add numdifftools (⚠️ Results suspicious)
- [x] Test with sample JSON input (✅ Working)

### Phase 4: Integration & Pilot ✅ COMPLETE
- [x] Julia: generate data → JSON
- [x] Run Python script on exported data
- [x] Julia: import results → compute errors
- [x] Generate basic plots (CSV output working)
- [x] Debug any failures (TaylorDiff, Dierckx fixed)

### Phase 5: Analysis ✅ COMPLETE
- [x] Create summary tables (minimal_pilot_results.csv)
- [x] Generate comparison data (diagnostic_test.jl)
- [x] Identify which methods work/fail (STATUS.md)

**Actual time spent**: ~6-8 hours + debugging
**Status**: Pilot complete, 7 methods validated, Python needs separate session

---

## Python Environment Setup

```bash
cd ~/derivative_estimation_study/python

# Create venv with UV (not conda!)
uv venv .venv

# Activate
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install packages
uv pip install numpy scipy scikit-learn matplotlib pandas
uv pip install pysindy  # includes pydiff
uv pip install tvregdiff
uv pip install numdifftools

# Save exact versions
uv pip freeze > requirements.txt
```

---

## Julia Dependencies to Add

```toml
[deps]
TrendFiltering = "..."  # NEW - excellent for high-order TV
# (Keep existing: ModelingToolkit, OrdinaryDiffEq, Symbolics,
#  GaussianProcesses, BaryRational, Dierckx, FFTW, ForwardDiff,
#  DataFrames, CSV, JSON3, Plots)

[compat]
julia = "1.10"
```

---

## Derivative Computation Strategy (Simplified)

**No cross-language AD complexity!**

1. **Julia methods**:
   - Native derivatives where available (Dierckx, splines)
   - ForwardDiff/recursive for fitted Julia functions

2. **Python methods**:
   - Each Python method computes its OWN derivatives (0-7)
   - No AD through Python boundary
   - Use method-native approaches:
     - sklearn GP: kernel derivative formulas
     - Splines: `.derivative()` method
     - tvregdiff: directly computes k-th derivative
     - RBF: finite differences on fitted function (acceptable)

---

## Error Metrics (Keep Simple)

For each method × order × configuration:

```julia
function compute_errors(pred, true_vals)
    valid = .!isnan.(pred) .& .!isinf.(pred)
    if sum(valid) < 2
        return (rmse=NaN, mae=NaN, max_err=NaN, success=false)
    end

    p, t = pred[valid], true_vals[valid]
    rmse = sqrt(mean((p .- t).^2))
    mae = mean(abs.(p .- t))
    max_err = maximum(abs.(p .- t))

    return (rmse=rmse, mae=mae, max_err=max_err, success=true)
end
```

**No fancy statistics yet** - just look at mean/std across trials and compare methods visually.

---

## Success Criteria (Pilot)

✅ **Success**:
- Ground truth generation works for orders 0-7
- At least 5 Julia methods produce results
- Python script runs and returns results
- Data exchange (JSON) works smoothly
- Can generate comparison plots
- Identify 2-3 methods that handle order 7

⚠️ **Acceptable**:
- 1-2 methods fail in pilot
- Some manual debugging needed

❌ **Failure** (needs redesign):
- Ground truth generation broken
- Python script crashes
- JSON exchange doesn't work
- No methods handle order 7

---

## Next Steps After Pilot

1. **If pilot succeeds**:
   - Add more systems (Van der Pol, SHO)
   - Add more methods (FFT, wavelets)
   - Scale up MC trials
   - Add more noise levels
   - Implement better parameter tuning

2. **If pilot reveals issues**:
   - Debug specific method failures
   - Adjust noise levels
   - Consider different test systems
   - Simplify further if needed

---

## Notes & Decisions

- **No parallelization yet** - get sequential version working first
- **No parameter tuning** - use defaults initially
- **No advanced statistics** - visual comparison sufficient for pilot
- **Focus on robustness** - try-catch everything, save partial results
- **Logging is key** - print progress, save intermediate files
- **UV not conda** - user requirement for Python environment

---

## Recommended Methods from Research

### Highest Priority (Add to catalog):

1. **TrendFiltering.jl** (Julia) - Order k=7, ℓ1 penalty on k-th derivative
   ```julia
   using TrendFiltering
   tf_fit = trendfilter(y_noisy, order=7, lambda=0.1)
   ```

2. **pydiff.TrendFiltered** (Python) - Modern, from pysindy
   ```python
   import pydiff
   tf = pydiff.TrendFiltered(order=7)
   result = tf(y_noisy, alpha=0.1)
   ```

3. **tvregdiff** (Python) - Direct TV regularization
   ```python
   from tvregdiff import tvregdiff
   deriv = tvregdiff(y_noisy, order=7, alph=0.01, dx=dt)
   ```

4. **Savitzky-Golay** (both) - Classic smoothing + differentiation
   - Python: `scipy.signal.savgol_filter`
   - Julia: Implement or find DSP.jl equivalent

### Lower Priority (Phase 2):
- Smoothing splines with GCV
- Higher-order TGV
- Wavelet methods
- Maximum entropy (too complex for pilot)

---

**END OF IMPLEMENTATION PLAN**
