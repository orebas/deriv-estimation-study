import sys
import json
from pathlib import Path
import numpy as np

# Adjust the path to find the project's modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from scipy.optimize import minimize


def run_gp_test(x_train, y_train, kernel, test_name, restarts=5, normalize_y=False, alpha=0.0, optimizer="fmin_l_bfgs_b"):
    """Fits a GP model with the given kernel and prints the learned hyperparameters."""
    print(f"--- Running Test: {test_name} ---")
    try:
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=restarts,
            alpha=alpha,
            normalize_y=normalize_y,
            optimizer=optimizer,
        )

        gp.fit(x_train.reshape(-1, 1), y_train)

        print("Learned Kernel Hyperparameters:")
        print(f"  {gp.kernel_}")
    except Exception as e:
        print(f"ERROR during test '{test_name}': {e}")
    print("-" * 40 + "\n")


def powell_then_lbfgsb(obj_func, initial_theta, bounds):
    # Powell (derivative-free) coarse search
    f = lambda th: obj_func(th)[0]
    res1 = minimize(f, initial_theta, method="Powell", bounds=bounds,
                    options={"maxiter": 2000, "xtol": 1e-4, "ftol": 1e-4})
    # L-BFGS-B refinement with gradients
    f2 = lambda th: obj_func(th)
    res2 = minimize(lambda th: f2(th)[0], res1.x,
                    jac=lambda th: f2(th)[1],
                    method="L-BFGS-B", bounds=bounds,
                    options={"maxiter": 1000, "ftol": 1e-9})
    val, _grad = f2(res2.x)
    return res2.x, float(val)


def main():
    """Load data and compare GP fits with different kernel bounds and optimizers."""
    # Use absolute path to input for robustness
    input_file = str(Path(__file__).resolve().parent / "bad_run_input.json")

    with open(input_file, 'r') as f:
        data = json.load(f)

    x_train = np.array(data["times"])
    y_train = np.array(data["y_noisy"])
    y_true = np.array(data["ground_truth_derivatives"]["0"])  # clean signal

    x_std = np.std(x_train) if np.std(x_train) > 0 else 1.0
    length_scale_init = float(x_std / 8.0)
    amp_init = float(np.var(y_train))

    # 1) Baselines (as before)
    original_kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) \
                     + WhiteKernel(noise_level=1e-8, noise_level_bounds=(1e-10, 1e-2))
    fixed_kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) \
                   + WhiteKernel(noise_level=1e-8, noise_level_bounds=(1e-10, 1e1))

    run_gp_test(x_train, y_train, original_kernel, "Original Narrow Bounds", restarts=5)
    run_gp_test(x_train, y_train, fixed_kernel, "Proposed Wide Bounds", restarts=5)

    # 2) Smart noise init and exact noise for reference
    try:
        noise_variance_guess = np.var(np.diff(y_train, n=2)) / 6.0
        exact_noise_variance = float(np.var(y_train - y_true))
        print(f"Smart noise variance (2nd-diff): {noise_variance_guess:.6f}")
        print(f"Exact noise variance (y_noisy - y_true): {exact_noise_variance:.6f}\n")
    except Exception as e:
        print(f"ERROR computing noise estimates: {e}")

    smart_init_kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) \
                        + WhiteKernel(noise_level=noise_variance_guess, noise_level_bounds=(1e-10, 1e1))
    run_gp_test(x_train, y_train, smart_init_kernel, "Smart Initialization + Wide Bounds", restarts=5)

    # 3) Better inits + normalize_y + jitter + more restarts
    better_init_kernel = ConstantKernel(amp_init, (1e-6, 1e6)) * RBF(length_scale=length_scale_init, length_scale_bounds=(1e-3, 1e3)) \
                         + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e1))
    run_gp_test(x_train, y_train, better_init_kernel, "Better Inits + normalize_y + jitter + restarts=30",
                restarts=30, normalize_y=True, alpha=1e-10)

    # 4) Powell → LBFGS-B optimizer
    wide_kernel_for_powell = ConstantKernel(amp_init, (1e-6, 1e6)) * RBF(length_scale=length_scale_init, length_scale_bounds=(1e-3, 1e3)) \
                              + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e1))
    run_gp_test(x_train, y_train, wide_kernel_for_powell, "Powell→LBFGSB + normalize_y + jitter", restarts=0,
                normalize_y=True, alpha=1e-10, optimizer=powell_then_lbfgsb)

    # 5) Same Powell pipeline but with x standardized
    try:
        x_mean = np.mean(x_train)
        xs = (x_train - x_mean) / x_std
        run_gp_test(xs, y_train, wide_kernel_for_powell, "Powell→LBFGSB + normalize_y + jitter (x standardized)", restarts=0,
                    normalize_y=True, alpha=1e-10, optimizer=powell_then_lbfgsb)
    except Exception as e:
        print(f"ERROR in standardized-x test: {e}")

    # 6) Two-stage: fix noise ~ exact, optimize amp/ell; then release noise and refine
    try:
        exact_fixed = ConstantKernel(amp_init, (1e-6, 1e6)) * RBF(length_scale=length_scale_init, length_scale_bounds=(1e-3, 1e3)) \
                      + WhiteKernel(noise_level=exact_noise_variance, noise_level_bounds=(exact_noise_variance*0.999, exact_noise_variance*1.001))
        print("Two-stage optimization:")
        print(" Stage 1: noise tightly bounded around exact")
        run_gp_test(x_train, y_train, exact_fixed, "Stage 1 (tight noise)", restarts=0, normalize_y=True, alpha=1e-10)

        # Extract amp/ell from Stage 1 result to seed Stage 2
        gp_stage1 = GaussianProcessRegressor(kernel=exact_fixed, n_restarts_optimizer=0, alpha=1e-10, normalize_y=True)
        gp_stage1.fit(x_train.reshape(-1,1), y_train)
        k1 = gp_stage1.kernel_
        amp1 = float(k1.k1.k1.constant_value) if hasattr(k1.k1, 'k1') else float(amp_init)
        ell1 = float(k1.k1.k2.length_scale) if hasattr(k1.k1, 'k2') else float(length_scale_init)

        stage2_kernel = ConstantKernel(amp1, (1e-6, 1e6)) * RBF(length_scale=ell1, length_scale_bounds=(1e-3, 1e3)) \
                        + WhiteKernel(noise_level=exact_noise_variance, noise_level_bounds=(1e-10, 1e1))
        print(" Stage 2: release noise bounds and refine all params")
        run_gp_test(x_train, y_train, stage2_kernel, "Stage 2 (refine)", restarts=20, normalize_y=True, alpha=1e-10,
                    optimizer=powell_then_lbfgsb)
    except Exception as e:
        print(f"ERROR in two-stage optimization: {e}")


if __name__ == "__main__":
    main()
