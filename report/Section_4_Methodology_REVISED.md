# 4. Methodology

## 4.1 Test System

We selected the **Lotka-Volterra predator-prey model** as our test system, a canonical benchmark in dynamical systems.

**System equations:**
```
dx/dt = αx - βxy    (prey dynamics)
dy/dt = δxy - γy    (predator dynamics)
```

where x(t) is the prey population and y(t) is the predator population.

**Parameters:**
- α = 1.5 (prey growth rate)
- β = 1.0 (predation rate)
- γ = 3.0 (predator death rate)
- δ = 1.0 (predator growth from predation)

**Initial conditions:**
- x₀ = 1.0 (prey)
- y₀ = 1.0 (predator)

**Time span:** [0, 10] with 101 equally-spaced points (Δt ≈ 0.1)

**Observable:** We tested derivative estimation on the **prey population x(t)**, which exhibits oscillatory behavior.

**Important limitation:** Only one variable (prey) from the two-state system was analyzed. Conclusions may not generalize even within the same system, as prey and predator trajectories have different oscillatory properties and noise sensitivities. Ideally, both variables should be evaluated.

**Ground truth generation:**

High-precision ground truth was generated using the following rigorous procedure:

1. **Symbolic differentiation:** Derivatives up to order 7 were computed symbolically using ModelingToolkit.jl's symbolic differentiation engine, yielding exact differential equations for each derivative order
2. **Augmented system:** The original Lotka-Volterra equations were augmented with these symbolic derivative equations to create an extended ODE system containing x(t), y(t), dx/dt, d²x/dt², ..., d⁷x/dt⁷ as state variables
3. **Numerical integration:** The augmented system was solved using the Vern9 algorithm (9th-order Verner method) with absolute tolerance 10⁻¹² and relative tolerance 10⁻¹²
4. **Validation:** Analytical derivative formulas for orders ≥3 in coupled nonlinear systems are intractable. Convergence was verified by comparing Vern9 solutions at tolerances 10⁻¹² vs 10⁻¹⁴ for orders 0-5, showing agreement to <10⁻¹⁰ relative error. Higher-order derivatives (6-7) are subject to greater numerical uncertainty and should be interpreted cautiously.

This approach avoids interpolant-based automatic differentiation and provides high-accuracy numerical ground truth constrained by ODE solver precision (validated to ~10⁻¹⁰ for orders 0-5).

**Sampling resolution concern:** With Δt ≈ 0.1, high-order derivatives (especially orders 6-7) approach the Nyquist limit for oscillatory signals. This coarse resolution may suppress some methods due to aliasing rather than algorithmic weakness. Future work should include convergence testing with finer grids.

**Rationale for system choice:** Lotka-Volterra provides:
- Nonlinear oscillatory dynamics representative of many scientific domains
- Symbolic differentiability enabling rigorous ground truth
- Moderate computational cost for extensive benchmarking

## 4.2 Noise Model

**Type:** Additive white Gaussian noise

**Scaling:** For each noise level ε, we added noise scaled by the clean signal's standard deviation:
```
y_noisy(t) = y_true(t) + ε · std(y_true) · η(t)
```
where η(t) ~ N(0, 1) is standard normal noise.

**Rationale:** Constant-variance Gaussian noise is a standard model for measurement error. Scaling by signal standard deviation ensures consistent interpretation across different signals.

**Important limitation:** Additive noise can produce negative population values, violating biological constraints. Multiplicative noise (proportional to signal magnitude) would be more realistic but was not tested. Results may not generalize to measurement models where noise is strictly positive or heteroscedastic.

**Noise levels tested:** Seven levels spanning approximately 7 orders of magnitude:
- 10⁻⁸ (near-noiseless)
- 10⁻⁶
- 10⁻⁴
- 10⁻³
- 10⁻² (1%)
- 2×10⁻² (2%)
- 5×10⁻² (5%)

**Interpretation:** For the Lotka-Volterra prey population with std(x) ≈ 0.29, noise level 10⁻² corresponds to absolute noise std ≈ 0.0029.

**Randomization:** Three independent noise realizations per configuration using Mersenne Twister PRNG with seeds 12345, 12346, 12347 for trials 1-3 respectively, ensuring exact reproducibility.

**Pseudo-replication caveat:** All three trials use the same underlying trajectory; variability reflects only the noise model, not dynamical diversity. A more robust design would test multiple initial conditions or parameter sets.

## 4.3 Experimental Design

**Configurations tested:**
- 8 derivative orders (0-7)
- 7 noise levels (10⁻⁸ to 5×10⁻²)
- 3 trials per configuration
- **Total:** 8 × 7 × 3 = 168 test cases per method

**Method coverage:**
- **Full coverage** (all 56 order×noise combinations): 16 methods
- **Partial coverage**: 11 methods
  - Central-FD, TVRegDiff-Julia: 14/56 configurations (orders 0-1 only - library provides stencils/regularization only up to 1st order)
  - Dierckx-5, ButterworthSpline_Python, Butterworth_Python, Whittaker_m2_Python, fourier, fourier_continuation, RKHS_Spline_m2_Python, KalmanGrad_Python, SVR_Python: 42/56 configurations (orders 0-5, missing 6-7 - degree/capability limits)

**Endpoint treatment:** First and last evaluation points excluded from all error computations to avoid boundary effects, leaving 99 interior points for analysis.

**Important note:** Excluding edge points removes regions where many practical estimators exhibit boundary degradation. Metrics therefore evaluate performance on an idealized interior sub-problem and may overstate accuracy for applications requiring full-domain estimates.

**Failure handling:**
- Methods returning NaN or Inf for >80% of evaluation points marked as failed for that configuration
- Failed configurations excluded from that method's statistics
- Partial failures documented separately

**Data pipeline:**
1. Generate ground truth for Lotka-Volterra system once
2. For each configuration (noise level × trial):
   - Add noise to ground truth prey trajectory
   - Export to JSON for Python methods
   - Evaluate all Julia methods in-process
   - Call Python script with 300s timeout
   - Collect results (predictions, timing, success status)
3. Aggregate results across trials

## 4.4 Evaluation Metrics

### 4.4.1 Root Mean Squared Error (RMSE)

**Definition:**
```
RMSE = sqrt(mean((y_pred - y_true)²))
```

Computed over interior points only (indices 2 to n-1).

**Limitation:** RMSE values are not comparable across derivative orders because derivative magnitudes vary by orders of magnitude (e.g., std(x) ≈ 0.3 vs std(d⁷x/dt⁷) ≈ 10⁻⁴).

### 4.4.2 Mean Absolute Error (MAE)

**Definition:**
```
MAE = mean(|y_pred - y_true|)
```

**Advantage:** Robust to outliers

**Limitation:** Same cross-order comparison issue as RMSE

### 4.4.3 Normalized RMSE (nRMSE) — Primary Metric

**Definition:**
```
nRMSE = RMSE / std(y_true)
```

**Critical clarification:** The standard deviation in the denominator is computed **for the specific derivative order being evaluated**, not the base signal. Thus:
- For order 0: nRMSE = RMSE(x) / std(x)
- For order 1: nRMSE = RMSE(dx/dt) / std(dx/dt)
- For order 7: nRMSE = RMSE(d⁷x/dt⁷) / std(d⁷x/dt⁷)

This makes nRMSE order-comparable: a value of 0.2 means "error is 20% of typical variation in that specific derivative" regardless of order.

**Interpretation:** Error expressed as a fraction of signal variation

**Performance thresholds** (calibrated via visual inspection of derivative plots):
- < 0.1: Visually accurate reconstruction
- 0.1 - 0.3: Moderate deviation visible
- 0.3 - 1.0: Substantial error but structure recognizable
- > 1.0: Error exceeds typical signal variation

**Note:** These thresholds are empirical guidelines, not rigorously validated cutoffs. Readers should interpret absolute nRMSE values directly.

**Justification:** Absolute metrics (RMSE, MAE) favor low-order derivatives where magnitudes are smaller. Normalized metrics enable fair comparison across orders, revealing which methods handle noise amplification effectively.

**Zero-crossing robustness:** Normalization uses std(y_true) computed over the same 99 interior evaluation points as RMSE (not including endpoints), avoiding division-by-zero when derivatives cross zero while maintaining consistency between numerator and denominator domains.

## 4.5 Statistical Analysis and Limitations

**Central tendency:** Mean nRMSE across 3 trials reported as primary statistic

**Uncertainty quantification:**
- Standard deviation across 3 trials
- 95% confidence intervals computed using t-distribution with df=2

**CRITICAL LIMITATION - Insufficient Statistical Power:**

With only n=3 trials per configuration:
- 95% CI half-width = t₀.₉₇₅,₂ × SD / √3 ≈ 2.48 × SD (extremely wide with df=2)
- **No formal hypothesis tests performed** due to insufficient sample size
- Claims of method superiority are **exploratory**, not statistically definitive
- CI overlap/non-overlap provides **suggestive evidence only**, not formal significance

**Interpretation guidelines:**
- Non-overlapping CIs across methods → **moderate evidence** of difference (not "strong" or "statistically significant")
- Overlapping CIs → **insufficient evidence** to claim difference (NOT "no difference")
- Consistent ordering across all 3 trials → **robust ranking pattern**

**Rationale for n=3:** Computational cost (24 methods × 168 configs × median 0.5s = ~5 hours total) was balanced against basic repeatability verification. This is sufficient to detect catastrophic failures and gross performance differences, but **inadequate for fine-grained method ranking** or quantifying small effect sizes.

**Pseudo-replication:** Trials differ only in random noise seed, not in underlying dynamics. Variability reflects noise model, not biological or parameter diversity.

**Recommended interpretation:** Treat rankings as **descriptive summaries** of performance on this specific system, not as statistically validated general statements. Methods that differ by <2× in nRMSE should be considered comparable given sample size.

**Future work:** Testing on 10+ diverse signals (varied ICs, parameters, systems) with ≥10 trials each is needed for robust statistical inference (see Section 10.2).

## 4.6 Hyperparameter Optimization Protocol

**Objective:** Provide equivalent optimization effort to all tunable methods.

**Gaussian Processes:**
- Hyperparameters: length scale (ℓ), signal variance (σ²_f), noise variance (σ²_n)
- Optimization: Maximum Likelihood Estimation (MLE) via L-BFGS-B
- Implementation: GaussianProcesses.jl (Julia), scikit-learn (Python)
- Bounds: ℓ ∈ [0.01, 10], σ² ∈ [0.01, 10] (scaled to problem's time domain and signal variance)
- Initialization: 3 random restarts (seeded deterministically per trial: seed + restart_idx) to mitigate local minima while ensuring reproducibility

**Splines:**
- Smoothing parameter (λ) selected via Generalized Cross-Validation (GCV)
- Automatic per-dataset tuning
- Implementation: Dierckx.jl (Julia), scipy.interpolate (Python)

**AAA Rational Approximation:**
- Tolerance: 10⁻¹³ (greedy termination criterion)
- Fixed for all evaluations (non-tunable)
- Precision: BigFloat (256-bit) for AAA-HighPrec, Float64 for AAA-LowPrec

**Fourier Spectral:**
- Filter fraction: 0.4 (retains lower 40% of frequency spectrum)
- **Optimization procedure:** Preliminary grid search over [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95] on a **separate validation run** (Lotka-Volterra with noise=10⁻³, orders 0-7), selecting value minimizing mean rank
- Fixed at 0.4 for all reported test results

**Methodological concern:** Unlike GP (per-dataset MLE) and splines (per-dataset GCV), the Fourier filter fraction was pre-tuned and fixed. This may provide an advantage relative to methods without access to validation data. However, 0.4 is a reasonable default for oscillatory signals and represents typical practitioner usage. Alternative: implement per-dataset tuning via cross-validation (adds computational cost).

**Total Variation (TVRegDiff):**
- Regularization parameter (α) auto-tuned via internal algorithm
- Iteration limit: 100, convergence tolerance: 10⁻⁶

**Finite Differences:**
- No tunable parameters (stencil determined by order and point count)

**Computational budget:** 300 second timeout per evaluation

## 4.7 Experimental Controls and Data Integrity

### 4.7.1 Exclusion Criteria

Methods were excluded from final analysis if they met any condition below. **Important caveat:** The specific criteria were established during initial data exploration, not pre-registered before data collection. This creates potential for apparent cherry-picking.

**Criteria:**

1. **Cross-language implementation failure:** When the same algorithm has implementations in both Julia and Python, and one shows >50× worse mean nRMSE after parameter parity verification
2. **Numerical breakdown:** Mean nRMSE > 10⁶ across all configurations
3. **Coverage failure:** Method fails (NaN/Inf) on >80% of test configurations

**Note on criterion 1:** The 50× threshold is pragmatic but arbitrary. Language ecosystems naturally differ; ideally both implementations should be debugged to parity. We document both results where feasible (see Section 4.7.2).

### 4.7.2 Methods Excluded

**Full candidate list:** 27 methods initially evaluated (see Appendix A for complete list)

**Excluded (3 methods):**

**1. GP-Julia-SE** (Gaussian Process with Squared Exponential kernel):
- Mean nRMSE: 38,238,701 (catastrophic numerical failure)
- Likely cause: Hyperparameter optimization collapse (length scale → 0 or → ∞) or kernel derivative implementation error
- Decision: Exclude; functional GP alternatives exist (GP-Julia-AD, GP_RBF_Python)
- **Transparency note:** Implementation may be debuggable; exclusion based on observed failure, not theoretical limitation

**2. TVRegDiff_Python** (Total Variation Regularized Differentiation):
- Mean nRMSE: 14.186 (72× worse than Julia implementation: 0.195)
- **Parameter parity checks performed:**
  - Regularization α matched across languages
  - Boundary conditions verified (periodic vs natural)
  - Iteration limits synchronized (100 max)
  - Convergence tolerances matched (10⁻⁶)
- Decision: Exclude Python version; retain Julia version (TVRegDiff-Julia)
- **Transparency note:** Despite parity efforts, discrepancy persists; may reflect subtle algorithmic differences not captured by parameter matching

**3. SavitzkyGolay_Python:**
- Mean nRMSE: 15,443 (17,500× worse than Julia: 0.881)
- Likely cause: Despite attempts to match window size and polynomial degree parameters, performance remained drastically inferior, suggesting implementation differences beyond parameter tuning
- Decision: Exclude Python version; retain Julia version (Savitzky-Golay)

**Exclusion impact:** Final analysis includes 24 of 27 candidates. Exclusions documented as implementation quality findings (Section 7.6).

### 4.7.3 Coverage Accounting

**Full coverage** (56/56 configs): 16 methods

**Partial coverage:** 11 methods (see list in Section 4.3)

**Ranking policy:**
- Overall rankings computed only over configurations where methods were tested
- Coverage percentages reported in all ranking tables
- **Naive rankings are misleading:** Methods tested only on easy configurations (low orders/noise) appear artificially superior
- **Fair comparison:** Primary rankings restricted to 16 full-coverage methods

### 4.7.4 Runtime Measurement Standardization

**Hardware:** <!-- TODO: Fill from comprehensive_summary.csv or system info -->
- CPU: AMD/Intel <!-- specific model -->
- RAM: 32 GB
- OS: Linux 6.6.87.2 (WSL2)

**Precision consistency:**
- Timing measured at same numerical precision as accuracy runs
- AAA-HighPrec: BigFloat for both
- All others: Float64

**Timing procedure:**
1. Exclude data preprocessing (JSON I/O, array allocation)
2. Time only: model fitting + derivative evaluation
3. Julia methods: 1 warm-up run to exclude JIT compilation
4. Report median across 3 trials

**Timeout:** 300 seconds → method marked as failed

### 4.7.5 Endpoint and Boundary Handling

**Evaluation grid:** All methods evaluated on identical 99 interior points (indices 2:100)

**Boundary treatment (method-specific):**
- **Fourier:** Symmetric extension for periodicity (note: Lotka-Volterra trajectories are oscillatory but not strictly periodic; this assumption may introduce edge artifacts despite interior-only evaluation)
- **Splines:** Natural boundary conditions (d²y/dx²=0 at endpoints)
- **Finite differences:** Stencils shrink at boundaries; edges excluded
- **GP:** No special treatment (kernel extrapolates naturally)

**Consistency check:** Verified all methods produce predictions on same grid before error computation

### 4.7.6 Statistical Validity

Covered in Section 4.5 (merged to eliminate redundancy).

## 4.8 Software and Implementation

**Julia environment:**
- Version: 1.9.3
- Key packages:
  - DifferentialEquations.jl 7.9.1
  - GaussianProcesses.jl 0.12.5
  - FFTW.jl 1.7.1
  - Dierckx.jl 0.5.3
  - ModelingToolkit.jl, Symbolics.jl (ground truth)

**Python environment:**
- Version: 3.11.5
- Key packages: numpy 1.25.2, scipy 1.11.2, scikit-learn 1.3.1

**Code availability:**
- Repository: <!-- TODO: Add GitHub URL and commit hash -->
- Zenodo archive: <!-- TODO: Add DOI upon publication -->
- License: MIT
- Includes: All source code, configuration files, raw results CSV, figure generation scripts

**Reproducibility provisions:**
- Fixed random seeds ensure exact noise realization reproducibility
- Environment specifications: `Project.toml` (Julia), `requirements.txt` (Python)
- Docker container: <!-- TODO: Decide if providing, else remove line -->

**Full workflow:** `src/comprehensive_study.jl` orchestrates all steps; see README for invocation.
