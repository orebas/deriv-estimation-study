# Paper Progress & Decision Log
**Last Updated:** 2025-10-20

## Current Status

### ✅ Completed
1. **Data Tables** (supplementary_data_tables_final.pdf - 25 pages)
   - Part I: Tables S1-S8 (Method × Noise, one per derivative order)
   - Part II: Tables M1-M16 (Noise × Order, one per full-coverage method)

2. **Abstract** - Written and reviewed

3. **Supplementary Plots** - 44 LINEAR scale plots with adaptive capping
   - Location: `paper_figures/supplementary_linear/`
   - Using: RED-YELLOW-GREEN colormap (intuitive)
   - Cap: 2.0 for easy conditions (order ≤3 AND noise ≤1e-4), 5.0 for hard
   - 6 plot sets covering all perspectives
   - **Currently excludes:** AAA-HighPrec, AAA-LowPrec, SavitzkyGolay_Python, GP-Julia-SE

4. **PDF Compilation** - Tested and working

### 🔄 In Progress / Pending Decisions

#### IMMEDIATE DECISION: Filter SpectralTaper_Python?
- **Evidence:** Poor performance at orders 4-7 (nRMSE 5-28)
- **Issue:** Polluting visualizations by pushing color scale
- **Recommendation:** YES - add to EXCLUDED list
- **Action needed:** Regenerate 44 plots with this exclusion

#### DISCOVERED: Chebyshev nRMSE Decreases with Order
- **Pattern:** Order 5→6→7: nRMSE goes 3.54 → 2.56 → 1.48 (DECREASES!)
- **Explanation:** RMSE grows (28k → 260k → 2.3M) but true derivative norm grows FASTER
- **Implication:** nRMSE is NOT order-comparable (normalization changes meaning)
- **Impact on paper:** Need to discuss this carefully, maybe use RMSE instead?
- **Uniqueness:** Only Chebyshev shows this because:
  - High-degree global polynomial (deg=20)
  - No numerical differentiation noise amplification
  - Constant absolute error across derivatives

### 📝 Still TODO
1. **Write Section 2: Related Work**
2. **Populate references.bib** with key citations
3. **Hyperparameter Optimization** (NEW - see below)

---

## Plot Generation History

### Evolution
1. **First attempt:** 20 comprehensive plots, log10 scale, all methods
   - Problem: "going up to 10 smushes everything to the bottom"

2. **Filtered version:** Excluded 4 problematic methods, still log10
   - Problem: "taking means over noise levels is kind of nonsense"

3. **Granular version:** 46 plots, NO averaging, still log10
   - Problem: "we shouldn't need to take logs of nrmse"

4. **CURRENT:** 44 LINEAR scale plots with adaptive capping ✅
   - Location: `paper_figures/supplementary_linear/`
   - Script: `generate_linear_plots.py`

### Next Iteration (Pending Decision)
5. **Proposed:** Same as current but exclude SpectralTaper_Python
   - Would reduce from 23 to 22 methods
   - Should improve color scale clarity

---

## Methods Summary

### Total Methods: 27
- **Full-coverage (all 56 conditions):** 13 methods
- **After current filtering:** 23 methods shown in plots
- **After proposed filtering:** 22 methods (remove SpectralTaper_Python)

### Excluded from Plots
1. **AAA-HighPrec** - Catastrophic failure at high orders (nRMSE > 10^10)
2. **AAA-LowPrec** - Less interesting variant, also fails
3. **SavitzkyGolay_Python** - Cross-language issues
4. **GP-Julia-SE** - Failed experiment, bad code
5. **SpectralTaper_Python** (PROPOSED) - Poor performance, pollutes visualizations

---

## Key Insights from Exploratory Analysis

### 1. Chebyshev Behavior
- Uses degree-20 polynomial (capped from potential deg=n_train-1)
- Analytic derivatives via `.deriv(m=order)`
- **Unique property:** nRMSE decreases with order (only method showing this)
- **Root cause:** Constant absolute error ÷ exponentially growing true derivative

### 2. SpectralTaper_Python Behavior
- FFT-based spectral differentiation with Tukey window
- Low-pass filtering + spectral shrinkage
- Works OK for orders 0-1, poor for 4-7
- **Verdict:** Not competitive, should exclude

### 3. nRMSE Metric Limitation
- **Within-order comparisons:** Valid and useful
- **Cross-order comparisons:** Misleading (normalization changes meaning)
- **Implication:** Rankings can reverse between orders
- **For paper:** Need to be careful about claims involving "overall best"

---

## Hyperparameter Questions (NEXT FOCUS)

### Methods with Hard-Coded Hyperparameters

#### 1. Chebyshev
- **Current:** `deg = max(3, min(20, n_train - 1))`
- **Question:** Is 20 optimal? Should it adapt to noise/order?
- **Note:** This IS regularization, not just a safety cap

#### 2. Fourier Methods
- **Current:** Various hard-coded percentages (40%, etc.)
- **Question:** What exactly is at 40%? Modes kept? Frequency cutoff?
- **Need to investigate:** Fourier implementation details

#### 3. SpectralTaper_Python
- **Current params:**
  - `SPEC_TAPER_ALPHA = 0.25` (Tukey window parameter)
  - `SPEC_CUTOFF = 0.5` (fraction of Nyquist)
  - `SPEC_SHRINK = 0.0` (spectral shrinkage 0-1)
- **Question:** Are these tuned? Can we improve them?

#### 4. Other Methods to Check
- GP methods: length scale, amplitude (currently fitted)
- Whittaker: lambda parameter (`WHIT_LAMBDA = 100.0`)
- RKHS: regularization parameters
- SVR: kernel parameters

### Investigation Plan
1. List all methods with hyperparameters
2. Identify which are hard-coded vs optimized
3. For hard-coded: check if they adapt to noise/order or are universal
4. Propose optimization strategy (cross-validation? grid search?)

---

## Open Questions

1. **Should we use RMSE instead of nRMSE for cross-order comparisons?**
   - Pro: No normalization weirdness
   - Con: Not comparable across different test functions

2. **Why does AAA-LowPrec do worse than Chebyshev?**
   - AAA should theoretically be more powerful (rational approximation)
   - Is it a hyperparameter issue?
   - Is it numerical precision?
   - Need to investigate

3. **What's the typical n_train in our experiments?**
   - Determines if degree=20 cap is even active
   - If n_train=100, we're using deg=20 (capped)
   - If n_train=15, we're using deg=14 (not capped)

4. **Should we have different hyperparameters for different (order, noise) regimes?**
   - E.g., higher Chebyshev degree for low noise?
   - Lower degree for high noise?

---

## Paper Structure (Planned)

### Main Paper
- Abstract ✅
- Introduction (TODO)
- Related Work (TODO)
- Methodology (TODO)
- Results (TODO)
- Discussion (TODO)
- Conclusion (TODO)

### Supplementary Materials
- Data Tables PDF ✅
- Linear Scale Plots (44 images) ✅
- Plot Catalog LaTeX (if needed)

---

## Workflow Notes

### Running Python Scripts
- **MUST** activate venv: `source ../venv/bin/activate`
- Reason: Numpy version conflicts with system Python

### Regenerating Plots
```bash
cd /home/orebas/derivative_estimation_study/report
source ../venv/bin/activate
python generate_linear_plots.py
```

### Data Source
- **File:** `../results/comprehensive/comprehensive_summary.csv`
- **Size:** 1,310 rows (27 methods × ~56 conditions, some missing)
- **Format:** method, category, language, deriv_order, noise_level, mean_nrmse, ...

---

## Next Steps (Priority Order)

1. **DECIDE:** Exclude SpectralTaper_Python? (YES recommended)
2. **IF YES:** Regenerate 44 plots with updated exclusion list
3. **INVESTIGATE:** Hyperparameters in Chebyshev and Fourier
4. **INVESTIGATE:** Why AAA-LowPrec underperforms Chebyshev
5. **INVESTIGATE:** Typical n_train values in experiments
6. **CONSIDER:** Hyperparameter optimization strategy
7. **WRITE:** Related Work section
8. **POPULATE:** references.bib
