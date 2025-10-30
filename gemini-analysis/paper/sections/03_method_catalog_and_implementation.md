# 4. Method Catalog and Implementation

This study did not begin with a fixed set of methods. Rather, it followed an exploratory research path, starting with methods known to work on noiseless data and progressively incorporating more sophisticated techniques in response to the challenges posed by noisy signals. This section provides a narrative of that journey, cataloging the methods investigated and detailing the significant implementation work required to create a fair and robust comparison.

### 4.1 A Composable Framework: Approximation + Automatic Differentiation

A core finding of our study is the power of a composable, two-step framework for numerical differentiation. Rather than relying on bespoke algorithms for each derivative order, the most successful methods, particularly Gaussian Process Regression, implicitly or explicitly follow a simple and highly effective recipe:

1.  **Fit an Approximant:** First, a smooth, analytic function (the "approximant") is fitted to the noisy data. This function can be a Gaussian Process, a spline, a spectral representation, or any other differentiable model. The goal of this step is to capture the underlying signal while filtering out noise.

2.  **Differentiate Analytically via AD:** Second, this fitted function is differentiated analytically using Automatic Differentiation (AD). Because the approximant is a well-defined mathematical function, AD can compute its derivatives to machine precision, free from the truncation and differencing errors that plague finite difference methods.

This "Approximant-AD" pattern is exceptionally powerful because it decouples the problem of noise-robust smoothing from the problem of differentiation. It allows practitioners to leverage the vast and mature ecosystems of both statistical modeling and automatic differentiation, effectively bringing the full power of modern AD frameworks to bear on a classical numerical problem.

### 4.2. Breadth of the Study

For practitioners, the choice of a specific software package can be as consequential as the choice of the underlying algorithm. To this end, our study intentionally includes multiple implementations of similar or identical algorithms (e.g., several variants of Gaussian Process Regression and Fourier-based methods) to assess whether real-world performance differences arise from implementation details. While we could not be exhaustive, we sought to include representatives from all major methodological families as well as several novel approaches.

### 4.3. A Narrative of Method Selection

Our investigation proceeded in three main waves:

1.  **Initial Foray: Adapting Noiseless Methods.** Our first attempts focused on adapting the successful AAA rational approximant method from our prior work. When these direct adaptations proved unstable on noisy data, we turned to a literature search for established techniques, leading us to methods based on local polynomial fitting (e.g., Savitzky-Golay, finite differences) and splines. While some of these showed moderate success, their effectiveness was often limited by the degree of the local polynomial, particularly for high-order derivatives.

2.  **Second Wave: Denoising and Statistical Approaches.** Recognizing that explicitly handling noise was critical, we next investigated two new classes of methods. The first class involved explicit denoising steps, such as those based on wavelet filtering (MAD) or spectral filtering, applied before differentiation. The second, more sophisticated class involved methods with a statistical foundation that inherently model noise, such as Gaussian Process Regression, Total Variation Regularization, and Kalman filtering.

3.  **Final Cohort.** The combination of these exploratory waves resulted in the final, comprehensive set of 42+ methods that form the basis of this benchmark.

### 4.4. Implementation: The Challenge of High-Order Derivatives

A significant practical challenge was that almost no off-the-shelf package natively supports the computation of arbitrarily high-order derivatives. Overcoming this required substantial implementation effort.

*   **Augmentation with Automatic Differentiation:** For methods whose underlying approximant was differentiable (e.g., Gaussian Processes), we augmented the implementation with Automatic Differentiation (AD) to compute the derivatives. Where possible, we used Taylor-mode AD, which is essential for the efficient computation of high-order derivatives, as naive nested AD has exponential complexity.

*   **Analytic Derivatives:** For other methods, such as those based on splines or Fourier series, we derived and implemented the analytical expressions for their higher-order derivatives directly.

*   **Noise Model Integration:** In some cases, we also had to implement custom noise-estimation steps to provide the algorithms with required hyperparameters, using techniques such as wavelet MAD or simple finite-difference-based noise estimation.

This substantial, and often non-trivial, implementation work was a prerequisite for creating the level playing field upon which this benchmark is built.
