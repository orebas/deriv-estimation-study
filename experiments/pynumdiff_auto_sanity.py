"""
PyNumDiff Auto (optimize + tvgamma) sanity experiment on a simple sine signal.

- Builds a clean sine + small noise signal
- Estimates tvgamma via docs heuristic from an estimated cutoff_frequency
- Runs optimize() for several PyNumDiff methods
- Compares against conservative tuned/fallback settings

Run:
  python/.venv/bin/python experiments/pynumdiff_auto_sanity.py
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks

from pynumdiff import smooth_finite_difference as sfd
from pynumdiff import total_variation_regularization as tvr
from pynumdiff import kalman_smooth
from pynumdiff.optimize import optimize as pnd_optimize


@dataclass
class Signal:
    t: np.ndarray
    x: np.ndarray
    dx: np.ndarray
    dddx: np.ndarray


def build_sine_signal(n: int = 2000, duration_s: float = 2.0, freq_hz: float = 2.0, noise_std: float = 1e-3) -> Signal:
    t = np.linspace(0.0, duration_s, n, dtype=float)
    dt = float(t[1] - t[0])
    x_clean = np.sin(2.0 * math.pi * freq_hz * t)
    # First derivative of sine
    dx_clean = 2.0 * math.pi * freq_hz * np.cos(2.0 * math.pi * freq_hz * t)
    x_noisy = x_clean + np.random.default_rng(12345).normal(0.0, noise_std, size=n)
    omega = 2.0 * math.pi * freq_hz
    d3x_clean = - (omega ** 3) * np.cos(omega * t)
    return Signal(t=t, x=x_noisy, dx=dx_clean, dddx=d3x_clean)


def estimate_cutoff_frequency(x: np.ndarray, t: np.ndarray) -> float:
    # Prefer robust peak counting
    duration = float(t[-1] - t[0]) if len(t) > 1 else 1.0
    if duration <= 0:
        duration = len(t) * (t[1] - t[0] if len(t) > 1 else 1.0)
    scale = float(np.std(x)) or float(np.max(x) - np.min(x) + 1e-9)
    peaks, _ = find_peaks(x, prominence=max(1e-3 * scale, 1e-9))
    if duration > 0 and len(peaks) > 0:
        return max(1.0 / duration, float(len(peaks)) / duration)

    # Fallback to FFT 90% cumulative power
    dt = float(np.mean(np.diff(t))) if len(t) > 1 else 1.0
    yf = np.fft.fft(x - np.mean(x))
    freqs = np.fft.fftfreq(len(x), d=dt)
    power = np.abs(yf) ** 2
    idx = np.argsort(power)[::-1]
    csum = np.cumsum(power[idx])
    i90 = int(np.searchsorted(csum, 0.9 * csum[-1]))
    fc = abs(float(freqs[idx[i90]]))
    fnyq = 0.5 / dt
    return max(0.01 * fnyq, min(fc, 0.3 * fnyq))


def tvgamma_from_cutoff(cutoff_hz: float, dt: float) -> float:
    # As per PyNumDiff docs (2b notebook)
    log_gamma = -1.6 * np.log(cutoff_hz) - 0.71 * np.log(dt) - 5.1
    return float(np.exp(log_gamma))


def nrmse(y_hat: np.ndarray, y_true: np.ndarray, trim_frac: float = 0.1) -> float:
    y_hat = np.asarray(y_hat)
    y_true = np.asarray(y_true)
    n = y_true.size
    # Trim symmetric ends (interior-only) to reduce endpoint blow-up
    k = int(max(1, min(n // 4, round(trim_frac * n))))  # cap to keep enough points
    sl = slice(k, max(k, n - k))
    y_hat_i = y_hat[sl]
    y_true_i = y_true[sl]
    m = np.isfinite(y_hat_i) & np.isfinite(y_true_i)
    if m.sum() < 3:
        return float("nan")
    err = y_hat_i[m] - y_true_i[m]
    denom = float(np.std(y_true_i[m]) or 1.0)
    return float(np.sqrt(np.mean(err * err)) / denom)


def derivative_k_from_x(x_hat: np.ndarray, t: np.ndarray, k: int) -> np.ndarray:
    # Smooth differentiation via cubic/quintic spline, then k-th derivative evaluated on t
    try:
        spline = UnivariateSpline(t, x_hat, k=5, s=0.0)
    except Exception:
        spline = UnivariateSpline(t, x_hat, k=3, s=0.0)
    return spline.derivative(n=k)(t)


def run_method_auto(sig: Signal) -> None:
    dt = float(np.mean(np.diff(sig.t)))
    cutoff = estimate_cutoff_frequency(sig.x, sig.t)
    tvg = tvgamma_from_cutoff(cutoff, dt)

    print("=== Settings ===")
    print(f"dt={dt:.6g}, cutoff_est={cutoff:.6g} Hz, tvgamma={tvg:.6g}")

    def _run_auto(func, name: str) -> None:
        params, val = pnd_optimize(func, sig.x, dt, tvgamma=tvg, padding='auto', maxiter=20)
        x_hat, dx_hat = func(sig.x, dt, **params)
        n_auto = nrmse(dx_hat, sig.dx)
        print(f"AUTO  {name:28s} nRMSE={n_auto:.6g} params={params}")

    print("\n=== Smooth-then-differentiate (first derivative) ===")
    _run_auto(sfd.butterdiff,   "butterdiff")
    _run_auto(sfd.gaussiandiff, "gaussiandiff")
    _run_auto(sfd.friedrichsdiff, "friedrichsdiff")
    try:
        params, val = pnd_optimize(sfd.splinediff, sig.x, dt, tvgamma=tvg, padding='auto', maxiter=20)
        xh, dxh = sfd.splinediff(sig.x, dt, **params)
        print(f"AUTO  {'splinediff':28s} nRMSE={nrmse(dxh, sig.dx):.6g} params={params}")
    except Exception as e:
        print(f"AUTO  {'splinediff':28s} FAILED: {e}")

    print("\n=== Smooth-then-differentiate (third derivative) ===")
    # butterdiff -> x_hat, then 3rd derivative via spline
    params, _ = pnd_optimize(sfd.butterdiff, sig.x, dt, tvgamma=tvg, padding='auto', maxiter=20)
    xh, _ = sfd.butterdiff(sig.x, dt, **params)
    d3 = derivative_k_from_x(xh, sig.t, 3)
    print(f"AUTO  {'butterdiff→jerk':28s} nRMSE={nrmse(d3, sig.dddx):.6g} params={params}")

    params, _ = pnd_optimize(sfd.gaussiandiff, sig.x, dt, tvgamma=tvg, padding='auto', maxiter=20)
    xh, _ = sfd.gaussiandiff(sig.x, dt, **params)
    d3 = derivative_k_from_x(xh, sig.t, 3)
    print(f"AUTO  {'gaussiandiff→jerk':28s} nRMSE={nrmse(d3, sig.dddx):.6g} params={params}")

    params, _ = pnd_optimize(sfd.friedrichsdiff, sig.x, dt, tvgamma=tvg, padding='auto', maxiter=20)
    xh, _ = sfd.friedrichsdiff(sig.x, dt, **params)
    d3 = derivative_k_from_x(xh, sig.t, 3)
    print(f"AUTO  {'friedrichsdiff→jerk':28s} nRMSE={nrmse(d3, sig.dddx):.6g} params={params}")

    try:
        params, _ = pnd_optimize(sfd.splinediff, sig.x, dt, tvgamma=tvg, padding='auto', maxiter=20)
        xh, _ = sfd.splinediff(sig.x, dt, **params)
        d3 = derivative_k_from_x(xh, sig.t, 3)
        print(f"AUTO  {'splinediff→jerk':28s} nRMSE={nrmse(d3, sig.dddx):.6g} params={params}")
    except Exception as e:
        print(f"AUTO  {'splinediff→jerk':28s} FAILED: {e}")

    print("\n=== Total Variation (first derivative direct) ===")
    params, val = pnd_optimize(tvr.velocity, sig.x, dt, tvgamma=tvg, padding='auto', maxiter=20)
    xh, dxh = tvr.velocity(sig.x, dt, **params)
    print(f"AUTO  {'tv.velocity':28s} nRMSE={nrmse(dxh, sig.dx):.6g} params={params}")

    print("\n=== Total Variation (third derivative direct) ===")
    params, val = pnd_optimize(tvr.jerk, sig.x, dt, tvgamma=tvg, padding='auto', maxiter=20)
    xh, d3h = tvr.jerk(sig.x, dt, **params)
    print(f"AUTO  {'tv.jerk':28s} nRMSE={nrmse(d3h, sig.dddx):.6g} params={params}")

    print("\n=== Kalman RTS (first derivative direct) ===")
    params, val = pnd_optimize(kalman_smooth.rtsdiff, sig.x, dt, tvgamma=tvg, padding='auto', maxiter=20)
    xh, dxh = kalman_smooth.rtsdiff(sig.x, dt, **params)
    print(f"AUTO  {'kalman.rtsdiff':28s} nRMSE={nrmse(dxh, sig.dx):.6g} params={params}")

    print("\n=== Kalman RTS (third derivative via spline) ===")
    d3 = derivative_k_from_x(xh, sig.t, 3)
    print(f"AUTO  {'kalman.rtsdiff→jerk':28s} nRMSE={nrmse(d3, sig.dddx):.6g} params={params}")


def main() -> None:
    sig = build_sine_signal(n=4000, duration_s=2.0, freq_hz=2.0, noise_std=5e-4)
    run_method_auto(sig)


if __name__ == "__main__":
    main()


