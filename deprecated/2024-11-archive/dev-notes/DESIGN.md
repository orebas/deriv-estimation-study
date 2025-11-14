# High-Order Derivative Estimation Study: Comprehensive Design Document

**Version:** 1.0 (ORIGINAL DESIGN)
**Date:** 2025-10-13
**Author:** Claude & User

---

## ⚠️ DOCUMENT STATUS

**This is the ORIGINAL comprehensive design document from the planning phase.**

**For current project status, see:**
- **STATUS.md** - Current implementation status, working methods, diagnostic results
- **IMPLEMENTATION_PLAN.md** - Simplified plan with current phase progress

**Key Changes from Original Design:**
1. **Python Integration**: Using standalone scripts + JSON/CSV (NOT PyCall as originally planned)
2. **TaylorDiff API**: Requires `Val(n)` syntax, not integer `n`
3. **Namespace Issues**: Multiple packages export `derivative`, must qualify with package name
4. **Working Methods**: 7 of 15 methods validated (AAA, Dierckx, scipy RBF/splines)
5. **Known Failures**: GP-Julia-SE (external C lib), Fourier-Interp (numerical explosion)

See STATUS.md for complete findings and diagnostic test results.

---

## Executive Summary

This study aims to **compare 15-20 "out-of-the-box" numerical differentiation methods** for estimating **high-order derivatives (5th, 6th, 7th)** from ODE trajectory data. The focus is on **comparative evaluation** rather than method optimization, testing implementations from Julia, Python, and potentially other languages.

### Key Design Decisions
- **Two separate studies**: Noiseless (ideal case) and Noisy (realistic case)
- **Pragmatic implementation**: Try all methods, abandon those that are "too damn hard"
- **Pilot phase**: 1-2 MC trials to verify all methods work
- **Full phase**: Scale up after pilot succeeds
- **Multi-language**: Julia primary, Python via PyCall

---

## 1. Method Catalog (Target: 15-20 methods)

### Category 1: Gaussian Process Regression (Expected: GOOD for high orders)

| # | Method | Package | Language | Difficulty | Priority |
|---|--------|---------|----------|------------|----------|
| 1 | GP-Julia-SE | GaussianProcesses.jl | Julia | EASY (done) | HIGH |
| 2 | GP-Python-sklearn | scikit-learn | Python | MEDIUM | HIGH |
| 3 | GP-Python-GPy | GPy | Python | MEDIUM | MEDIUM |
| 4 | GP-Python-george | george | Python | MEDIUM | LOW |

**Notes**: Already have #1 working. Test different kernel implementations (SE, Matern, RBF).

### Category 2: Rational Approximation (Expected: VARIABLE)

| # | Method | Package | Language | Difficulty | Priority |
|---|--------|---------|----------|------------|----------|
| 5 | AAA-HighPrec | BaryRational.jl (tol=1e-14) | Julia | EASY (done) | HIGH |
| 6 | AAA-LowPrec | BaryRational.jl (tol=0.1) | Julia | EASY (done) | HIGH |
| 7 | AAA-BIC | Auto BIC selection | Julia | EASY (done) | HIGH |
| 8 | FHD-8 | Floater-Hormann deg 8 | Julia | EASY | MEDIUM |

**Notes**: All implemented in `bary_derivs.jl`. Test with `nth_deriv_at()` for derivatives.

### Category 3: Spectral/Fourier Methods (Expected: SOMETIMES WORKS)

| # | Method | Package | Language | Difficulty | Priority |
|---|--------|---------|----------|------------|----------|
| 9 | FFT-Spectral | FFTW.jl | Julia | MEDIUM | HIGH |
| 10 | Chebyshev | ApproxFun.jl or FastChebInterp.jl | Julia | HARD | MEDIUM |
| 11 | Fourier-Interp | Custom (in bary_derivs.jl) | Julia | EASY (done) | MEDIUM |

**Notes**: FFT needs periodic extension handling. Chebyshev may be complex to implement correctly.

### Category 4: RBF Methods (Expected: MAY/MAY NOT WORK)

| # | Method | Package | Language | Difficulty | Priority |
|---|--------|---------|----------|------------|----------|
| 12 | RBF-Multiquadric | scipy.interpolate.Rbf | Python | EASY | HIGH |
| 13 | RBF-Gaussian | scipy.interpolate.Rbf | Python | EASY | HIGH |
| 14 | RBF-ThinPlate | scipy.interpolate.Rbf | Python | EASY | MEDIUM |

**Notes**: scipy makes this easy. Test different basis functions.

### Category 5: Spline Methods (Expected: FAIL - negative controls)

| # | Method | Package | Language | Difficulty | Priority |
|---|--------|---------|----------|------------|----------|
| 15 | Dierckx-5 | Dierckx.jl (k=5) | Julia | EASY (done) | HIGH |
| 16 | Scipy-Cubic | scipy.interpolate.CubicSpline | Python | EASY | HIGH |
| 17 | Scipy-Univariate | scipy.interpolate.UnivariateSpline | Python | EASY | MEDIUM |

**Notes**: Expected to fail for 7th derivatives due to low local degree. Include as baselines.

### Category 6: Regularization Methods (Expected: OK to GOOD)

| # | Method | Package | Language | Difficulty | Priority |
|---|--------|---------|----------|------------|----------|
| 18 | TV-Differentiation | Custom or Python library | TBD | HARD | HIGH |
| 19 | Tikhonov-Reg | Custom implementation | Julia | MEDIUM | MEDIUM |

**Notes**: TV differentiation - need to find/implement. Tikhonov needs parameter tuning.

### Category 7: Finite Differences & Others (Expected: FAIL - baselines)

| # | Method | Package | Language | Difficulty | Priority |
|---|--------|---------|----------|------------|----------|
| 20 | Central-FD | Custom | Julia | EASY | HIGH |
| 21 | Python-numdifftools | numdifftools | Python | EASY | MEDIUM |
| 22 | LOESS | Loess.jl | Julia | EASY (done) | LOW |

**Notes**: FD expected to fail catastrophically. Include for comparison.

---

## 2. Implementation Architecture

### File Structure
```
derivative_estimation_study/
├── DESIGN.md                      # This file
├── src/
│   ├── comprehensive_study.jl     # Main driver (NEW)
│   ├── methods/
│   │   ├── julia_methods.jl       # All Julia implementations
│   │   ├── python_methods.jl      # PyCall wrappers
│   │   ├── spectral_methods.jl    # FFT, Chebyshev, etc.
│   │   └── regularization.jl      # TV, Tikhonov
│   ├── experiments/
│   │   ├── ground_truth.jl        # Symbolic derivative generation
│   │   ├── noise_models.jl        # Add noise to data
│   │   └── evaluation.jl          # Error metrics
│   └── analysis/
│       ├── visualize.jl           # Plotting functions
│       └── statistics.jl          # Summary tables
├── results/
│   ├── pilot/                     # Pilot study results (1-2 trials)
│   └── full/                      # Full study results
├── figures/
│   ├── pilot/
│   └── full/
└── paper/
    ├── manuscript.tex
    └── supplementary.tex
```

### Method Interface Design

All methods must conform to this interface:

```julia
"""
Abstract type for all differentiation methods
"""
abstract type DifferentiationMethod end

"""
Required interface for each method:
- fit_method(x, y, options) -> fitted_function
- evaluate_derivative(fitted_function, x_eval, order) -> y_deriv
- get_method_name() -> String
- get_method_category() -> String
"""

struct MethodResult
    name::String
    category::String
    predictions::Dict{Int, Vector{Float64}}  # order => values
    timing::Float64
    success::Bool
    error_message::Union{String, Nothing}
end
```

### Python Integration Strategy

```julia
using PyCall

# At module initialization
function __init__()
    # Import Python packages
    try
        @pyimport sklearn.gaussian_process as skgp
        @pyimport scipy.interpolate as interp
        @pyimport numpy as np
        global py_sklearn = skgp
        global py_scipy = interp
        global py_numpy = np
    catch e
        @warn "Python packages not available" exception=e
    end
end

# Wrapper example
function fit_sklearn_gp(x, y; kernel="RBF")
    gp = py_sklearn.GaussianProcessRegressor(
        kernel=py_sklearn.kernels.RBF(),
        alpha=1e-6,
        n_restarts_optimizer=5
    )
    gp.fit(reshape(x, :, 1), y)

    # Return Julia-callable function
    return xi -> begin
        pred = gp.predict(reshape([xi], 1, 1), return_std=false)
        return pred[1]
    end
end
```

---

## 3. Experimental Design

### 3.1 Test System

**Lotka-Volterra Predator-Prey Model:**
```
dx/dt = α*x - β*x*y
dy/dt = δ*x*y - γ*y

Parameters: α=1.5, β=1.0, γ=3.0, δ=1.0
Initial conditions: x(0)=1.0, y(0)=1.0
Time interval: [0, 10]
Observable: x(t) (prey population)
```

**Why Lotka-Volterra?**
- Nonlinear, oscillatory dynamics
- Smooth solutions amenable to high-order differentiation
- Well-understood behavior
- Can compute exact derivatives symbolically

**Alternative systems to consider (optional):**
- Van der Pol oscillator (stiffer)
- SIR epidemic model (different timescales)

### 3.2 Experimental Parameters

#### **PILOT STUDY** (to verify all methods work)
- **Monte Carlo trials**: 2
- **Noise levels**: [0%, 1%] (noiseless + one noisy case)
- **Data sizes**: [51, 101] (two densities)
- **Derivative orders**: 0 through 7
- **Random seed**: 12345

**Total pilot experiments**: 2 trials × 2 noise × 2 sizes × ~20 methods = **~160 runs**
**Estimated time**: 30-60 minutes

#### **FULL STUDY** (after pilot succeeds)

**Study A: NOISELESS CASE** (ideal)
- **Monte Carlo trials**: 5 (verify reproducibility, check numerical stability)
- **Noise level**: 0%
- **Data sizes**: [21, 51, 101, 201] (sparse to dense)
- **Derivative orders**: 0 through 7
- **Focus**: Which methods can theoretically handle high orders?

**Study B: NOISY CASE** (realistic)
- **Monte Carlo trials**: 30
- **Noise levels**: [0.1%, 0.5%, 1%, 2%] (low noise for high-order derivatives)
- **Data sizes**: [51, 101, 201] (dense data required)
- **Derivative orders**: 0 through 7
- **Focus**: Which methods are robust to noise?

**Total full experiments**:
- Noiseless: 5 × 1 × 4 × 20 = 400 runs
- Noisy: 30 × 4 × 3 × 20 = 7,200 runs
- **Total: ~7,600 runs**

**Estimated time**: 4-8 hours depending on method speed

### 3.3 Ground Truth Generation

Use symbolic differentiation via ModelingToolkit.jl:

```julia
using ModelingToolkit, Symbolics

function compute_exact_derivatives(system, observable, max_order=7)
    @named sys = system
    eqs = equations(sys)
    eq_dict = Dict(eq.lhs => eq.rhs for eq in eqs)

    derivs = Vector{Num}(undef, max_order+1)
    derivs[1] = observable  # 0th derivative

    for order in 1:max_order
        derivs[order+1] = substitute(
            expand_derivatives(D(derivs[order])),
            eq_dict
        )
    end

    return derivs
end
```

Then solve augmented ODE system with high precision:
```julia
prob = ODEProblem(augmented_system, ic, tspan, params)
sol = solve(prob, AutoVern9(Rodas4P()),
            abstol=1e-12, reltol=1e-12,
            saveat=time_points)
```

### 3.4 Noise Model

Additive Gaussian noise proportional to signal magnitude:

```julia
function add_noise(signal, noise_level, rng)
    σ = noise_level * mean(abs.(signal))
    noise = σ * randn(rng, length(signal))
    return signal + noise
end
```

**Rationale**:
- Noise level = 0.01 means 1% of mean signal magnitude
- Physically realistic for measurement noise
- Preserves signal structure

---

## 4. Error Metrics

For each method, derivative order, and configuration, compute:

### Primary Metrics
1. **RMSE** (Root Mean Square Error):
   ```
   RMSE = sqrt(mean((y_pred - y_true)^2))
   ```

2. **Relative RMSE**:
   ```
   Rel_RMSE = RMSE / std(y_true)
   ```
   - Normalizes by signal variability
   - Comparable across different derivative orders

3. **MAE** (Mean Absolute Error):
   ```
   MAE = mean(abs(y_pred - y_true))
   ```

4. **Max Error**:
   ```
   Max_Error = max(abs(y_pred - y_true))
   ```

### Secondary Metrics
5. **Success Rate**: % of MC trials where Rel_RMSE < 1.0 (or another threshold)

6. **Computational Cost**: Wall-clock time for fitting + evaluation

7. **Failure Analysis**: Count of NaN/Inf in predictions

### Aggregation
- Report **mean ± std** across MC trials
- Create **ranking tables** for each derivative order
- Generate **heatmaps**: methods × orders × noise levels

---

## 5. Analysis & Visualization Plan

### 5.1 Key Figures (Publication Quality)

#### **Figure 1: Method Comparison Overview**
- Multi-panel grid: rows = derivative orders (0,1,2,3,4,5,6,7), cols = noise levels
- Each panel: bar chart of mean Rel_RMSE for top 10 methods
- Highlights: best method per order, failure modes

#### **Figure 2: Error Scaling with Derivative Order**
- Log-log plot: derivative order vs RMSE
- One line per method (different colors/styles)
- Separate plots for noiseless vs noisy cases
- Shows exponential error growth

#### **Figure 3: Data Density Effects**
- Plot: # data points vs RMSE for orders 5, 6, 7
- Compare methods' sensitivity to data density
- Identify "dense data requirements"

#### **Figure 4: Example Trajectory**
- Top panel: Original signal + noisy observations + best fit
- Middle panels: Derivatives 1-3 (truth vs estimates)
- Bottom panels: Derivatives 5-7 (truth vs estimates)
- Demonstrates degradation visually

#### **Figure 5: Method Category Heatmap**
- Rows: method categories (GP, Rational, Spectral, RBF, Spline, Reg)
- Cols: derivative orders (5, 6, 7)
- Color: mean Rel_RMSE (green=good, red=bad)
- Quick visual summary

#### **Figure 6: Computational Cost vs Accuracy**
- Scatter plot: time vs RMSE for order 7
- Pareto frontier analysis
- Identifies efficient methods

### 5.2 Tables

#### **Table 1: Method Summary**
| Method | Category | Language | Orders 5-7 Mean RMSE | Rank |
|--------|----------|----------|----------------------|------|
| ... | ... | ... | ... | ... |

#### **Table 2: Noiseless Performance** (orders 5, 6, 7)
| Method | Order 5 | Order 6 | Order 7 |
|--------|---------|---------|---------|
| ... | ... | ... | ... |

#### **Table 3: Noisy Performance** (1% noise, orders 5, 6, 7)
| Method | Order 5 | Order 6 | Order 7 | Success Rate |
|--------|---------|---------|---------|--------------|
| ... | ... | ... | ... | ... |

#### **Table 4: Computational Cost**
| Method | Fit Time (ms) | Eval Time (ms) | Total (ms) |
|--------|---------------|----------------|------------|
| ... | ... | ... | ... |

### 5.3 Statistical Analysis

- **Friedman test**: Rank methods across multiple configurations
- **Pairwise comparisons**: Identify significantly different methods
- **Confidence intervals**: Error bars on all metrics
- **Correlation analysis**: Method similarity clustering

---

## 6. Implementation Phases

### **Phase 0: Setup & Verification** (Day 1, 2-3 hours)
- [ ] Set up fresh project structure
- [ ] Install all Julia packages
- [ ] Install Python packages (sklearn, scipy, GPy, numdifftools)
- [ ] Test PyCall integration
- [ ] Verify ground truth generation works
- [ ] Implement basic method interface

### **Phase 1: Julia Methods** (Day 1-2, 4-6 hours)
- [ ] Port from `bary_derivs.jl`:
  - [ ] AAA (high/low/BIC)
  - [ ] FHD
  - [ ] Fourier interpolation
  - [ ] GP (GaussianProcesses.jl)
- [ ] Implement new Julia methods:
  - [ ] FFT spectral differentiation
  - [ ] Chebyshev (if feasible)
  - [ ] Central finite differences
  - [ ] LOESS
  - [ ] Tikhonov regularization
- [ ] Test each method in isolation with `nth_deriv_at()`

### **Phase 2: Python Methods** (Day 2, 3-4 hours)
- [ ] sklearn GP wrapper
- [ ] scipy RBF (multiquadric, gaussian, thin-plate)
- [ ] scipy splines (cubic, univariate)
- [ ] numdifftools
- [ ] Test derivative computation via ForwardDiff/TaylorDiff

### **Phase 3: TV Differentiation** (Day 2-3, 2-4 hours)
- [ ] Research TV differentiation implementations
  - Python: `tvregdiff` package or custom
  - Julia: Custom implementation or port
- [ ] Test on sample data
- [ ] **Skip if too complex** (mark as "future work")

### **Phase 4: Pilot Study** (Day 3, 1-2 hours)
- [ ] Run pilot: 2 trials, 2 noise levels, 2 data sizes
- [ ] Verify all methods produce results
- [ ] Debug failures
- [ ] Generate pilot figures
- [ ] **Decision point**: Which methods to keep?

### **Phase 5: Full Study** (Day 3-4, 4-8 hours)
- [ ] Run noiseless study (Study A)
- [ ] Run noisy study (Study B)
- [ ] Monitor for crashes, NaNs, timeouts
- [ ] Save all results to CSV/JLS

### **Phase 6: Analysis** (Day 4-5, 4-6 hours)
- [ ] Generate all figures
- [ ] Create summary tables
- [ ] Run statistical tests
- [ ] Identify key findings

### **Phase 7: Paper Writing** (Day 5-6, 6-8 hours)
- [ ] Write methods section
- [ ] Write results section
- [ ] Write discussion
- [ ] Format figures and tables
- [ ] Compile LaTeX

**Total estimated time: 5-7 days of focused work**

---

## 7. Expected Findings (Hypotheses)

Based on user's prior experience:

### Hypothesis 1: GP Methods Excel
**Expected**: Gaussian Processes (especially with SE/RBF kernels) will perform best for orders 5-7 due to infinite smoothness assumption.

**Variation**: Different GP implementations may show subtle differences in hyperparameter optimization.

### Hypothesis 2: Rational Approximation is Variable
**Expected**: AAA will work well IF it has enough support points. Low-precision may fail for high orders.

**Key factor**: Effective degree of rational function.

### Hypothesis 3: Spectral Methods Are Fragile
**Expected**: FFT/Fourier work well for smooth periodic-like signals but fail with noise or non-periodic boundaries.

**Key factor**: Spectral leakage, endpoint handling.

### Hypothesis 4: RBF May/May Not Work
**Expected**: RBF with appropriate shape parameter might work, but "out-of-the-box" defaults likely fail.

**Key factor**: Shape parameter selection (often hard-coded poorly).

### Hypothesis 5: Splines Fail for High Orders
**Expected**: 5th-order splines cannot produce accurate 7th derivatives (not enough local polynomial degree).

**Confirmation**: These serve as negative controls.

### Hypothesis 6: TV Differentiation OK
**Expected**: TV regularization should handle moderate orders (5-6) by balancing smoothness and fidelity.

**Challenge**: Parameter tuning required.

### Hypothesis 7: Noise Amplification is Exponential
**Expected**: Error growth ~ O(noise * h^(-k)) for k-th derivative.

**Implication**: Even 1% noise may make 7th derivatives nearly impossible.

---

## 8. Potential Challenges & Mitigation

| Challenge | Likelihood | Impact | Mitigation |
|-----------|------------|--------|------------|
| Python package installation fails | Medium | High | Provide fallback Julia-only version |
| PyCall crashes or is unstable | Low | High | Isolate Python calls, catch exceptions |
| TV differentiation no good package | High | Medium | Skip it, mark as future work |
| Chebyshev too complex to implement | High | Low | Skip it, not critical |
| Some methods timeout | Medium | Medium | Set timeout (60s?), mark as failure |
| NaN/Inf in high-order derivatives | High | Medium | Detect and report, exclude from stats |
| Memory issues with dense data | Low | Medium | Use garbage collection, chunk processing |
| Results take > 12 hours | Low | High | Reduce MC trials or methods |

---

## 9. Code Quality & Reproducibility

### Requirements
- [x] **Random seed fixed**: 12345
- [ ] **All packages pinned**: Manifest.toml committed
- [ ] **No hard-coded paths**: Use `@__DIR__`
- [ ] **Logging**: Log all method attempts, successes, failures
- [ ] **Error handling**: Try-catch around each method, save error messages
- [ ] **Progress bars**: Show progress during long runs
- [ ] **Checkpointing**: Save intermediate results (every 10% progress)
- [ ] **Documentation**: Docstrings for all functions

### Reproducibility Checklist
```julia
# At top of main script
using Random, Serialization
Random.seed!(12345)

# Save environment
using Pkg
Pkg.instantiate()  # Ensures exact package versions

# Log system info
println("Julia version: ", VERSION)
println("System: ", Sys.MACHINE)
println("Date: ", Dates.now())

# Save all results
results = run_study(...)
serialize("results_$(Dates.format(now(), "yyyymmdd_HHMMSS")).jls", results)
```

---

## 10. Paper Outline

### Title
*Comparative Evaluation of Numerical Differentiation Methods for Estimating High-Order Derivatives from ODE Trajectory Data*

### Abstract (200 words)
- Motivation: High-order derivatives needed for system identification
- Problem: Most methods fail catastrophically
- Approach: Benchmark 15-20 methods on Lotka-Volterra
- Key findings: GP >> Rational > Spectral >> Splines (fail)
- Implications: Dense data + low noise essential, orders > 6 extremely challenging

### 1. Introduction
- Applications: ODE parameter estimation, data-driven modeling
- Challenge: Differentiation amplifies noise
- Prior work: Limited comparative studies on orders > 4
- Contributions: Systematic benchmark, multi-language implementations

### 2. Methods
- 2.1 Test Problem (Lotka-Volterra)
- 2.2 Ground Truth Generation (symbolic differentiation)
- 2.3 Noise Model
- 2.4 Method Catalog (categories, implementations)
- 2.5 Experimental Design (pilot + full)
- 2.6 Error Metrics

### 3. Results
- 3.1 Noiseless Case: Theoretical Performance
  - Figure 1, Table 1
- 3.2 Noisy Case: Realistic Performance
  - Figure 2, Table 2
- 3.3 Data Density Requirements
  - Figure 3
- 3.4 Computational Cost Analysis
  - Table 3
- 3.5 Example Visualization
  - Figure 4

### 4. Discussion
- 4.1 Why GP Methods Excel (infinite smoothness)
- 4.2 Limitations of Polynomial Methods (finite degree)
- 4.3 The Noise-Order Trade-off (exponential growth)
- 4.4 Practical Recommendations
- 4.5 When to Give Up (orders > 7)

### 5. Conclusions
- Best methods: GP (orders 5-7), Rational (orders 5-6)
- Worst methods: Splines, finite differences
- Takeaway: High-order differentiation is HARD
- Future work: Physics-informed methods, ensemble approaches

### Supplementary Material
- S1: Detailed method descriptions
- S2: Complete parameter tables
- S3: All figures for derivative orders 0-7
- S4: Code repository link

### Target Venues
- **Primary**: *Journal of Computational Physics*
- **Secondary**: *SIAM Journal on Scientific Computing*
- **Alternative**: *ACM Transactions on Mathematical Software* (more implementation-focused)

---

## 11. Success Criteria

### Pilot Study Success
✅ **Must achieve**:
- All Julia methods run without errors
- At least 50% of Python methods work
- Generate at least one complete set of results (2 trials × 2 configs)
- Plots render correctly

⚠️ **Acceptable**:
- 1-2 methods fail due to complexity
- Python methods slow but functional

❌ **Failure** (requires redesign):
- PyCall unusable
- >5 methods fail in pilot
- Ground truth generation broken

### Full Study Success
✅ **Must achieve**:
- Complete data for noiseless + noisy studies
- At least 12 methods with results for orders 5-7
- Statistical significance in method rankings
- Publication-ready figures

⚠️ **Acceptable**:
- 3-5 methods excluded due to failures
- Some methods only work up to order 6

❌ **Failure** (requires revision):
- <10 methods with complete results
- Numerical instabilities corrupt data
- No clear winner in comparisons

---

## 12. Post-Compaction Review Plan

After conversation compaction, present this design to:

1. **GPT-5** (via OpenAI API)
   - Focus: Methodological rigor, experimental design
   - Questions: Missing methods? Better test systems? Statistical tests?

2. **Gemini 2.5 Pro** (via Google AI)
   - Focus: Implementation feasibility, code architecture
   - Questions: Julia/Python integration issues? Performance concerns?

3. **Claude Opus 4** (via Anthropic API)
   - Focus: Paper structure, scientific communication
   - Questions: Results presentation? Discussion points?

**Consolidation**: Synthesize feedback, update design if major issues found.

---

## Appendix A: Package Requirements

### Julia Packages
```toml
[deps]
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
Serialization = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

# ODE solving
ModelingToolkit = "961ee093-0014-501f-94e3-6117800e7a78"
Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"

# Approximation methods
GaussianProcesses = "891a1506-143c-57d2-908e-e1f8e92e6de9"
BaryRational = "91aaffc3-5777-4842-85b7-5d3d5d6a3494"
Dierckx = "39dd38d3-220a-591b-8e3c-4c3a8c710a94"
Loess = "4345ca2d-374a-55d4-8d30-97f9976e7612"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
ApproxFun = "28f2ccd6-bb30-5033-b560-165f7b14dc2f"  # Optional

# Differentiation
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
TaylorDiff = "b36ab563-344f-407b-a36a-4f200bebf99c"

# Python integration
PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"

# Plotting
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"

# Optimization (for GP)
Optim = "429524aa-4258-5aef-a3af-852621145aeb"
LineSearches = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
```

### Python Packages (via pip/conda)
```bash
pip install numpy scipy scikit-learn matplotlib
pip install GPy george numdifftools  # Optional
# pip install tvregdiff  # If we find it
```

---

## Appendix B: Derivative Evaluation Strategy

Challenge: How to compute k-th derivative of fitted function?

### Strategy 1: Automatic Differentiation (Primary)
```julia
using ForwardDiff, TaylorDiff

# For k=1
deriv1(f, x) = ForwardDiff.derivative(f, x)

# For k>1, use TaylorDiff (faster)
derivk(f, x, k) = TaylorDiff.derivative(f, x, k)

# Fallback: recursive ForwardDiff (slower but works)
function nth_deriv_recursive(f, k, x)
    k == 0 && return f(x)
    k == 1 && return ForwardDiff.derivative(f, x)
    return ForwardDiff.derivative(t -> nth_deriv_recursive(f, k-1, t), x)
end
```

### Strategy 2: Analytical (when available)
```julia
# Splines: use built-in derivative function
Dierckx.derivative(spline, x, nu=k)

# FFT: multiply by (ik)^k in frequency domain
```

### Strategy 3: Numerical Finite Differences (last resort)
```julia
# Central difference approximation
function fd_deriv(f, x, k, h=1e-3)
    # Use numdifftools or custom implementation
end
```

---

## Appendix C: Contact & Maintenance

**Primary Contact**: User (orebas)
**Implementation Lead**: Claude (AI assistant)
**Timeline**: ~1 week intensive work
**Last Updated**: 2025-10-13

**Post-Study**:
- Code to be released as open-source (GitHub)
- Results data to be archived (Zenodo/Figshare)
- Paper preprint on arXiv before journal submission

---

**END OF DESIGN DOCUMENT**

Total pages: ~15
Total words: ~5,500
Estimated reading time: 25-30 minutes
