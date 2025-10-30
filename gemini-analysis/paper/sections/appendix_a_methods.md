# Appendix A: Method Descriptions & Experimental Setup

## A.1. Method Descriptions

*(This section would contain detailed mathematical descriptions of all 33 methods evaluated, similar to the content in the original `report/sections/section5_methods.tex`. This includes categories like Gaussian Processes, Spectral Methods, Splines, etc., with their formulations and key parameters.)*

... content to be ported from `section5_methods.tex` ...

## A.2. Experimental Setup

### A.2.1. ODE Test Systems

**1. Lotka-Volterra (Periodic):**
A two-dimensional system of autonomous ODEs used to model predator-prey dynamics.
$$
\begin{aligned}
\frac{dx}{dt} &= \alpha x - \beta xy \\
\frac{dy}{dt} &= \delta xy - \gamma y
\end{aligned}
$$
We use the prey population, $x(t)$, as our test signal.

**2. Lorenz System (Chaotic):**
A three-dimensional system known for its chaotic behavior.
$$
\begin{aligned}
\frac{dx}{dt} &= \sigma(y - x) \\
\frac{dy}{dt} &= x(\rho - z) - y \\
\frac{dz}{dt} &= xy - \beta z
\end{aligned}
$$
We use the $x(t)$ variable as our test signal.

**3. Van der Pol Oscillator (Periodic):**
A second-order non-linear oscillator with limit-cycle behavior.
$$
\frac{d^2x}{dt^2} - \mu(1 - x^2)\frac{dx}{dt} + x = 0
$$
We use the position, $x(t)$, as our test signal.

### A.2.2. Ground Truth Generation

Ground truth derivatives were generated to machine precision by augmenting each ODE system with its own symbolic derivatives and solving the extended system with a high-accuracy solver (Vern9, tolerance `1e-12`).

### A.2.3. Noise Model and Trials

Additive white Gaussian noise was added to the true signal, scaled by the signal's standard deviation. Seven noise levels were tested: `1e-8`, `1e-6`, `1e-4`, `1e-3`, `0.01`, `0.02`, and `0.05`. All results are the average of **10 trials**, each with a different random noise seed.

### A.2.4. Evaluation Metric

The primary metric used is the Normalized Root Mean Square Error (nRMSE), defined as:
$$
\text{nRMSE} = \frac{\sqrt{\text{mean}((\hat{f}^{(n)}(t) - f^{(n)}_{\text{true}}(t))^2)}}{\text{std}(f^{(n)}_{\text{true}})}
$$
This normalization makes the error comparable across different derivative orders.
