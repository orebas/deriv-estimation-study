# 4. Experimental Design

To create a robust and comprehensive benchmark, we first formally define the estimation problem.

### 4.1. Formal Problem Statement

**Given:**
*   A smooth but analytically unknown function $f: \mathbb{R} \to \mathbb{R}$.
*   A set of $n$ noisy observations $\{(t_i, y_i)\}_{i=1}^n$ where $y_i = f(t_i) + \epsilon_i$ on a uniform grid.
*   Noise is assumed to be additive and Gaussian, $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$.

**Objective:**
*   Estimate the $k$-th order derivative, $\hat{f}^{(k)}(t)$, for derivative orders $k \in \{0, 1, \ldots, 7\}$.

**Constraints:**
*   The evaluation is performed on the interior of the domain to avoid boundary effects that can unfairly penalize certain methods.
*   A method is considered to have failed a configuration if it does not produce a finite result.

### 4.2. Testbed: Dynamical Systems

We selected three well-known Ordinary Differential Equation (ODE) systems to serve as the source of our ground-truth data. These systems were chosen to represent a diversity of dynamic behaviors:

1.  **Lotka-Volterra:** A classic two-variable system exhibiting stable periodic oscillations.
2.  **Van der Pol Oscillator:** A system with a non-linear damping term that produces a stable limit cycle.
3.  **Lorenz System:** A three-variable system famous for its chaotic behavior, providing a more challenging test case.

For each system, trajectories were generated using a high-precision numerical integrator. The system's equations were then used to analytically compute the true derivatives up to the 7th order, providing a high-fidelity ground truth for our comparisons.

### 4.3. Noise Model

To simulate real-world data, we added Gaussian white noise to the integrated trajectories. The noise level was varied across a wide range, from `1e-8` (representing nearly clean data) up to `0.02` (representing a challenging 2% noise level relative to the signal's standard deviation). This wide range allows us to evaluate method performance in both high-precision, low-noise regimes and robustness-critical, high-noise regimes.

### 4.4. Evaluation Metrics

The primary metric for evaluation is the normalized Root-Mean-Square Error (nRMSE).

**Rationale for Normalization:** Direct comparison of raw RMSE values is not possible across different derivative orders because the magnitudes of the derivatives themselves vary dramatically. For instance, in the Lotka-Volterra system, the standard deviation of the true signal might be ~0.2, while the standard deviation of its seventh derivative can be ~10^5.

To create a fair, order-comparable metric, we normalize the raw RMSE by the standard deviation of the true derivative signal. This yields a dimensionless error metric where an nRMSE of 0.1 consistently means a 10% error relative to the signal's characteristic magnitude, regardless of the derivative order.

All metrics are computed after excluding the endpoints of the time series, as many methods exhibit boundary effects that can unfairly skew the results.

### 4.5. Success Criteria

While the nRMSE provides a continuous measure of performance, it is also useful to define a threshold for what constitutes a "successful" or "acceptable" result. A method is considered successful for a given configuration if it achieves an nRMSE of less than 1.0, meaning its average error is smaller than the typical magnitude of the true signal itself. An nRMSE greater than 1.0 indicates that the error is dominating the signal, and values greater than 10 are considered a catastrophic failure.
