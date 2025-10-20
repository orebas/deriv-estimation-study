# 6. Results

This section presents performance results for the 24 methods evaluated across 8 derivative orders (0-7) and 7 noise levels (10⁻⁸ to 5×10⁻²). All results are mean ± standard deviation across 3 trials and should be interpreted as descriptive summaries; statistical limitations (n=3, insufficient for formal hypothesis testing) are documented in Sections 4.5 and 6.7.

## 6.1 Overall Performance Rankings

**Figure 1** (Heatmap) visualizes method performance across derivative orders, showing mean nRMSE averaged over all noise levels for the top 15 methods. Methods are sorted by overall average performance (ascending). The heatmap uses log-scale coloring to accommodate the wide range of nRMSE values across different derivative orders.

**Key observations from Figure 1:**
- Clear performance stratification across derivative orders
- Most methods maintain relatively stable performance at orders 0-2
- Substantial degradation begins at orders 3-4 for many methods
- Only a subset of methods remain viable at orders 6-7

### 6.1.1 Full-Coverage Methods Ranking

To ensure fair comparison, **Table 2** ranks the 16 methods with full coverage (all 56 order×noise configurations tested).

**Table 2: Full-Coverage Methods Overall Performance**

| Rank | Method | Category | <!-- TODO: Add mean nRMSE from data --> | Coverage |
|------|--------|----------|-------------|----------|
| <!-- TODO: Generate ranking table from comprehensive_summary.csv filtered to full-coverage methods, sorted by mean nRMSE across all configs --> |

**Ranking methodology:** Mean nRMSE computed across all 56 configurations (8 orders × 7 noise levels), then averaged over 3 trials. Methods with partial coverage excluded from this ranking to avoid bias toward easy configurations (see Section 6.1.2).

<!-- TODO: Once ranking table is filled from data, add 1-2 sentences highlighting top 3 methods and any notable category patterns -->

### 6.1.2 Coverage Bias Analysis

**Critical limitation:** Methods with partial coverage appear artificially superior in naive overall rankings because they are only tested on configurations where they can succeed.

**Example:** TVRegDiff-Julia (14/56 configs, orders 0-1 only) and Central-FD (14/56 configs, orders 0-1 only) both achieve low nRMSE values—but only because they are excluded from the challenging high-order configurations where most methods struggle.

**Recommendation:** Use Table 2 (full-coverage methods only) for fair cross-method comparison. Partial-coverage methods should be evaluated within their tested scope.

## 6.2 Performance Across Derivative Orders

**Figure 2** (Small Multiples) presents an 8-panel grid showing nRMSE vs noise level for each derivative order (0-7). Each panel shows the top 7 full-coverage methods by overall mean nRMSE (consistent set across all panels; see Table 2 ranking), with mean ± standard deviation from 3 trials.

### 6.2.1 Derivative Order Progression

**Orders 0-1 (Function and First Derivative):**
- Most methods perform well in this regime (see Figure 2, orders 0-1 panels)
- Differentiation task is relatively easy; even simple methods succeed
- Performance differences between methods are modest

**Orders 2-3 (Second and Third Derivatives):**
- Clear separation emerges between method categories
- Some methods begin showing substantial degradation (nRMSE > 1.0) at noise ≥ 2%
- Gaussian Process and spectral methods maintain relatively stable performance

**Orders 4-5 (Fourth and Fifth Derivatives):**
- Extreme challenge for many methods
- Subset of methods exhibits catastrophic failure (nRMSE > 10)
- Only methods with sophisticated noise handling remain viable (nRMSE < 1.0)

**Orders 6-7 (Sixth and Seventh Derivatives):**
- Represents the most challenging test case
- Many methods fail completely (NaN/Inf or nRMSE >> 10)
- Limited subset of methods provides usable estimates even at moderate noise (10⁻²)
- Partial coverage is highest at these orders due to library/degree limitations

### 6.2.2 Method-Specific Observations

<!-- TODO: After reviewing Figure 2 data, add 2-3 specific observations about method behavior patterns, e.g.:
- Which method(s) show flat noise curves (robust) vs steep curves (sensitive)?
- At which order does each major category (GP, Rational, Spectral, Spline, FD) start degrading?
- Are there cross-over points where method rankings change with noise level?
-->

## 6.3 Qualitative Comparison: Visual Validation

**Figure 3** (Qualitative Comparison) shows actual derivative estimates for a challenging test case: 4th-order derivative at 2% noise level. Four methods are compared:
- GP-Julia-AD
- AAA-HighPrec
- Fourier-Interp
- <!-- TODO: Replace with a full-coverage method that performed poorly at order 4 (e.g., low-ranking spline or TrendFilter) - Central-FD only covers orders 0-1 per Section 6.6.1 -->

**Visual assessment:**
Each panel plots ground truth (black solid line) vs method prediction, with nRMSE value displayed. This visualization provides qualitative validation of the quantitative nRMSE metric.

**Insight:** nRMSE differences translate directly to visual reconstruction quality. Methods with nRMSE < 0.3 produce visually accurate derivatives that capture oscillatory structure; methods with nRMSE > 1.0 show substantial deviation from ground truth.

## 6.4 Accuracy vs Computational Cost Trade-Off

**Figure 4** (Pareto Frontier) plots mean computation time (x-axis, log scale) vs mean nRMSE (y-axis, log scale) for all 24 methods. Points are color-coded by category, with method names annotated.

### 6.4.1 Computational Complexity Observations

**Runtime distribution** (on our test system; see Section 4.7.5 for hardware details):
- Fast methods (< 0.01 s): Fourier/spectral methods, finite differences
- Moderate methods (0.01-0.1 s): Splines, some regularization methods
- Slow methods (> 0.1 s): Gaussian Processes, AAA-HighPrec, RKHS methods

**Scaling implications:**
- For n=101 points tested here, even slowest methods complete within reasonable time (median timings shown in Figure 4)
- At larger problem sizes (n > 1000), O(n³) methods (GPs) may become prohibitive
- O(n log n) spectral methods scale favorably for large-scale applications

### 6.4.2 Pareto-Optimal Methods

Methods on or near the Pareto frontier (no other method is both faster AND more accurate):
<!-- TODO: Identify Pareto-optimal methods from Figure 4 and list here -->

**Trade-off recommendations:**
- <!-- TODO: Based on Figure 4, provide guidance on method selection based on computational budget vs accuracy requirements -->

## 6.5 Performance vs Noise Level

**Figure 5** (Noise Sensitivity) shows nRMSE vs noise level curves for the top 5 full-coverage methods by overall ranking (Table 2), at selected derivative orders (0, 2, 4, 7). Error bars represent ± standard deviation across 3 trials.

### 6.5.1 Noise Robustness Patterns

**Low noise regime (≤ 10⁻⁴):**
- Most methods achieve excellent performance (nRMSE < 0.1) at orders 0-2
- Performance gap between methods is minimal
- Method choice less critical in this regime

**Moderate noise regime (10⁻³ to 10⁻²):**
- Substantial separation emerges, especially at orders ≥ 3
- Method selection becomes critical for accuracy
- Some methods maintain nRMSE < 0.5, others degrade to nRMSE > 2.0

**High noise regime (≥ 2%):**
- Extreme challenge even for best methods
- Only most robust methods maintain nRMSE < 1.0 at orders 0-2
- At orders ≥ 4, nearly all methods struggle (nRMSE > 1.0)

### 6.5.2 Noise Sensitivity Scaling

**Approximate noise scaling relationships** (empirical fit from Figure 5 data at order 4):

<!-- TODO: Fit and report approximate scaling relationships, e.g.:
- GP-Julia-AD: nRMSE ∝ noise^α (estimate α)
- AAA-HighPrec: nRMSE ∝ noise^β (estimate β)
- Note: These are empirical fits to this specific test system; scaling may differ for other signals
-->

## 6.6 Coverage and Failure Analysis

### 6.6.1 Method Coverage Summary

**Full coverage (56/56 configurations):** 16 methods
- Gaussian Processes (4), Rational (2), Spectral (5), Local Polynomial (1), Regularization (2), Other (2)

**Partial coverage:** 8 methods
- Orders 0-1 only (2 methods): Central-FD, TVRegDiff-Julia — library/implementation constraints
- Orders 0-5 only (6 methods): Dierckx-5, ButterworthSpline_Python, RKHS_Spline_m2_Python, fourier, fourier_continuation, others — degree/capability limits

### 6.6.2 Failure Mode Documentation

**Catastrophic failures** (nRMSE > 10⁶ or NaN/Inf):
- Occurred in <!-- TODO: Count configurations with catastrophic failure across all methods --> configurations

**Most common failure patterns:**
1. High-order derivatives (≥ 6) with high noise (≥ 2%)
2. Rational approximation methods at orders ≥ 3
3. Finite difference methods at any order with noise ≥ 10⁻³

**Failure causes** (inferred from implementation):
- Numerical instability in high-order rational function derivatives
- Noise amplification in finite difference stencils
- Ill-conditioning in kernel/covariance matrices
- Regularization parameter optimization collapse

## 6.7 Statistical Uncertainty

As documented in Section 4.5, n=3 trials provides insufficient power for formal hypothesis testing. The following interpretive guidelines apply to all results presented:

**Confidence intervals:** Where shown (Figures 2, 5), 95% CIs are extremely wide (half-width ≈ 2.48 × SD). Overlapping CIs do **not** imply "no difference"; non-overlapping CIs provide **moderate evidence** of difference, not statistical significance.

**Method rankings:** Treat as **exploratory descriptive summaries**, not definitive statistical statements. Methods differing by <2× in nRMSE should be considered comparable given sample size limitations.

**Robust patterns:** Findings that hold across all 3 trials (e.g., consistent method ordering, categorical performance differences exceeding 10×) represent more reliable patterns than specific numerical values.

## 6.8 Summary of Key Findings

1. **Derivative order is the primary difficulty driver:** Performance degrades systematically with increasing order across all methods and noise levels

2. **Method category stratification:** Clear performance hierarchy emerges by category, with Gaussian Processes and spectral methods generally outperforming splines, which outperform finite differences

3. **Coverage bias is substantial:** Partial-coverage methods appear artificially superior in naive rankings; fair comparison requires coverage normalization

4. **Noise amplification scales with order:** High-order derivatives (≥ 4) exhibit extreme noise sensitivity; even best methods struggle at noise ≥ 2%

5. **No universal best method:** Optimal choice depends on derivative order, noise level, and computational budget (Figure 4 Pareto frontier)

6. **Statistical power limitations:** Specific numerical rankings should be interpreted cautiously due to n=3 sample size

---

**Figures referenced:**
- **Figure 1:** Heatmap of method performance across derivative orders (report/paper_figures/publication/figure1_heatmap.png)
- **Figure 2:** Small multiples showing nRMSE vs noise for each order (report/paper_figures/publication/figure2_small_multiples.png)
- **Figure 3:** Qualitative comparison of derivative estimates (report/paper_figures/publication/figure3_qualitative.png)
- **Figure 4:** Pareto frontier of accuracy vs computational cost (report/paper_figures/publication/figure4_pareto.png)
- **Figure 5:** Noise sensitivity curves for top methods (report/paper_figures/publication/figure5_noise_sensitivity.png)
