# Appendix B: Visualizing the Fits

To provide a qualitative understanding of the performance differences, this section presents a visual comparison of a top-performing method (`GP-Julia-AD`) and a catastrophically failed method (`AAA-HighPrec`) on a challenging test case: the chaotic Lorenz system with 2% noise.

The figure below is a two-panel plot. The top panel shows how each method fits the noisy source data for the function itself (0th derivative). The bottom panel shows the corresponding estimate for the 3rd derivative, illustrating how errors are amplified.

![Lorenz System Fit Comparison](figures/lorenz_fit_comparison.png)

**Observations:**

*   In the top panel, the `GP-Julia-AD` fit (green line) successfully captures the underlying structure of the ground truth (black line) while smoothing through the noise (gray points). In contrast, the `AAA-HighPrec` fit (red dashed line) exhibits wild oscillations and deviates significantly from the data.
*   In the bottom panel, the `GP-Julia-AD` estimate for the 3rd derivative remains a stable and reasonably accurate approximation of the ground truth. The `AAA-HighPrec` estimate, however, has exploded to an entirely different order of magnitude, bearing no resemblance to the true signal. This visual evidence starkly illustrates the practical difference between a robust and an unstable method.
