# 4. Detailed Analysis: From Raw Data to Final Rankings

The journey from raw experimental output to the final summary table involved a deliberate, multi-stage filtering and analysis process. This section details the methodology used to distill the results and arrive at our final conclusions.

### 4.1. Initial Data Filtering: Removing Unviable Methods

The first step was to filter out methods that were fundamentally unsuitable or unstable for the task. This was not based on performance, but on viability.

1.  **Exclusion of Unstable Methods (AAA)**: Our implementations of the AAA (Adaptive Antoulas-Anderson) algorithm were found to be unstable for this specific task. Their errors were often orders of magnitude larger than any other method, and they frequently failed to produce valid output.

    It is important to frame this result: Our conclusion is not that rational approximants are unsuitable in general, but rather that a direct application does not appear to be robust to the combination of noise and differentiation.  The AAA algorithm is exceptionally powerful for function approximation in noise-free contexts.  We suspect that more sophisticated approaches, perhaps involving regularization or hybrid methods (e.g., fitting a rational approximant to a pre-smoothed signal), could unlock their potential. This remains a promising avenue for future research.

    Given these results, the AAA methods were removed from the main analysis cohort to prevent their extreme outliers from skewing aggregate statistics.

2.  **Exclusion of Incomplete Coverage**: A primary goal of this study was to evaluate methods across a wide range of derivative orders (0 through 7). Several methods, particularly some of the Python legacy methods and those based on low-degree splines, did not have full coverage across all orders or failed consistently on certain systems. To ensure a fair, apples-to-apples comparison in our final rankings, only methods that successfully produced results for all tested configurations were included in the "contender" set.

### 4.2. Defining the "Contender" Set

After this initial filtering, we were left with a set of 18 robust methods that provided full data coverage. This cohort became our "contender" set for the final analysis. All subsequent rankings and comparisons were performed *only within this group*. This is a crucial methodological point: the ranks presented in the summary table are ranks *among the contenders*, not among the initial, larger pool of all tested methods.

### 4.3. Ranking Methodology

To produce the final summary table, we followed a two-step ranking process:

1.  **Per-Cell Ranking**: For each individual experimental cell—defined by a unique combination of `(ODE_system, noise_level, derivative_order)`—we ranked the contender methods against each other based on their mean `nRMSE` (averaged across the 10 trials for that cell).

2.  **Averaging Ranks**: We then calculated the final "Avg. Rank" for each method by averaging these per-cell ranks across two distinct regimes:
    *   **Low-Noise Regime**: Averaged across noise levels `1e-8` and `1e-6`.
    *   **High-Noise Regime**: Averaged across noise levels `0.01` and `0.02`.

This methodology ensures that the final rank is a robust measure of a method's performance across a wide variety of conditions, and it prevents a single outlier or a particularly favorable test case from dominating the results. 

### 4.4. Quantitative Performance Metrics

To add further quantitative rigor to our analysis, we defined two specific metrics to characterize method performance beyond the simple average error.

*   **Noise Robustness:** We define a method's robustness for a given derivative order as the highest noise level at which its `nRMSE` remains at or below 0.1. A higher value on this metric indicates a method can maintain accuracy under more significant noise conditions. This provides a more concrete measure of robustness than average error alone.

*   **High-Order Stability:** To quantify how gracefully a method's performance degrades as the task becomes more difficult, we measure its high-order stability. This is defined as the slope of the log of the `nRMSE` versus the derivative order, calculated in the low-noise regime. A lower, flatter slope indicates that the method is more stable and its error grows more slowly as it is tasked with computing higher-order derivatives.

These metrics provide a complementary view to the main rankings and are used to inform the specific recommendations and trade-offs discussed in our conclusion.

### 4.6. Analysis of Coverage Bias

A critical aspect of a fair benchmark is understanding "coverage bias." Many methods are not designed to compute high-order derivatives. For example, `Central-FD` and `TVRegDiff-Julia` in our study only support up to order 1. If we were to rank all methods naively based on their average error across all the tests they passed, these methods would appear artificially superior, as they would only be evaluated on "easy" low-order configurations and would be exempted from the challenging high-order tests where most methods struggle.

To create a fair comparison, we therefore restricted our main summary table to the cohort of "contender" methods that successfully produced results for all tested configurations up to order 5. Methods with partial coverage are not inherently worse, but they should be considered specialists. A practitioner who only needs a first-order derivative might find that `TVRegDiff-Julia` is an excellent choice, but it cannot be fairly compared in a general-purpose ranking against a method that successfully computes 7th-order derivatives.

### 4.7. Performance Degradation by Derivative Order

A clear pattern emerging from the data is the systematic degradation of performance as the derivative order increases. This is an expected consequence of the ill-posed nature of differentiation. We can characterize this trend in several phases:

*   **Orders 0-2 (Low-Order):** In this regime, most contender methods perform well, and the performance differences between them are relatively modest, particularly in low-noise scenarios. The task of smoothing or finding a first or second derivative is not challenging enough to create significant separation between the top methods.

*   **Orders 3-5 (Mid-Order):** This is the regime where a clear separation emerges. The task becomes significantly more challenging, and methods without sophisticated noise handling begin to struggle. The performance of GPR and the stronger spectral methods remains high, while simpler spline- and filter-based methods see a substantial drop in accuracy.

*   **Orders 6-7 (High-Order):** This regime represents an extreme challenge. Only a very small subset of methods, primarily GPR, are able to produce a usable estimate, and even their errors are significant. For most other methods, the error in this regime constitutes a catastrophic failure. Derivative order is clearly the dominant factor in the difficulty of the estimation problem.

### 4.8. The Critical Role of Taylor-Mode AD for High-Order Derivatives

A key technical factor in the performance of high-order derivative estimation is the underlying mechanism of the Automatic Differentiation library used. Naively composing first-order AD operations (i.e., nested forward- or reverse-mode AD) to compute a high-order derivative results in an algorithm with exponential complexity, rendering it infeasible for orders beyond a handful.

The success of the `GP-Julia-AD` method, for instance, is critically dependent on its use of Taylor-mode AD. This mode is specifically designed to compute high-order derivatives efficiently by propagating a full Taylor series expansion through the computation, rather than just a single derivative value. This approach reduces the computational complexity significantly (often to polynomial time), making the computation of 5th, 6th, and 7th order derivatives tractable. This highlights that for practitioners seeking high-order derivatives, the choice of AD implementation is as important as the choice of the approximant itself.

### 4.9. Practitioner's Guide: Speed vs. Accuracy Trade-offs

The Pareto front plot in our summary of findings (Figure 4) provides a clear guide for practitioners to select a method based on their specific needs. Based on that analysis, we offer the following explicit recommendations:

*   **For Accuracy-Critical Applications:** When the highest possible accuracy is the primary concern and computational cost is secondary, a Gaussian Process method is the unequivocal best choice. Our results show `GP-Julia-AD` and the Python GPR variants consistently provide the lowest error.

*   **For Balanced Requirements:** For applications that require a strong balance between accuracy and speed, a spectral method is likely the optimal choice. `Fourier-Continuation` and `Fourier-Interp` provide accuracy that is competitive with GPR, but with a 10-20x speedup.

*   **For Speed-Critical Applications:** When computational speed is the dominant constraint (e.g., in real-time or large-scale applications), simpler methods are required. `Savitzky-Golay` offers a robust and extremely fast baseline, providing reasonable accuracy for a very low computational cost. The fastest spectral methods (e.g., `fourier`) are also excellent choices in this regime. It is also important to note that for very large datasets ($N > 1000$), the cubic scaling of GPR methods may become computationally prohibitive, making fast spectral or filtering methods the only viable options.
