"""
Intelligent method selection for PyNumDiff based on signal characteristics
Based on comprehensive testing results
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy import signal, fft


class SignalAnalyzer:
    """Analyze signal characteristics to guide method selection"""

    @staticmethod
    def analyze(y: np.ndarray, dt: float) -> Dict[str, any]:
        """
        Analyze signal properties to determine best differentiation method

        Returns dict with:
        - has_polynomial_trend: bool
        - polynomial_degree: float (estimated)
        - has_oscillations: bool
        - dominant_frequencies: list
        - noise_level: float (estimated)
        - is_smooth: bool
        - has_discontinuities: bool
        - boundary_importance: float (0-1)
        """
        n = len(y)
        t = np.arange(n) * dt

        # Detrend to separate polynomial from oscillations
        coeffs = np.polyfit(t, y, min(3, n//10))
        trend = np.polyval(coeffs, t)
        detrended = y - trend

        # Polynomial trend detection
        trend_strength = np.var(trend) / (np.var(y) + 1e-10)
        has_polynomial_trend = trend_strength > 0.3

        # Estimate polynomial degree by fitting increasing orders
        if has_polynomial_trend:
            poly_degree = SignalAnalyzer._estimate_polynomial_degree(t, y)
        else:
            poly_degree = 0

        # Oscillation detection via FFT
        freqs = fft.fftfreq(n, dt)
        fft_vals = np.abs(fft.fft(detrended))
        fft_vals[0] = 0  # Remove DC component

        # Find dominant frequencies
        threshold = 0.1 * np.max(fft_vals)
        dominant_freq_indices = np.where(fft_vals[:n//2] > threshold)[0]
        dominant_frequencies = freqs[dominant_freq_indices].tolist()
        has_oscillations = len(dominant_frequencies) > 0

        # Noise estimation (using high-frequency content)
        high_freq_power = np.mean(fft_vals[n//4:n//2])
        total_power = np.mean(fft_vals[:n//2])
        noise_level = high_freq_power / (total_power + 1e-10)

        # Smoothness check (using second differences)
        d2y = np.diff(y, 2)
        roughness = np.std(d2y) / (np.std(y) + 1e-10)
        is_smooth = roughness < 0.1

        # Discontinuity detection
        dy = np.diff(y)
        jump_threshold = 3 * np.std(dy)
        has_discontinuities = np.any(np.abs(dy) > jump_threshold)

        # Boundary importance (how much signal varies near boundaries)
        boundary_size = max(5, n // 20)
        boundary_var = np.var(y[:boundary_size]) + np.var(y[-boundary_size:])
        interior_var = np.var(y[boundary_size:-boundary_size])
        boundary_importance = boundary_var / (boundary_var + interior_var + 1e-10)

        return {
            'has_polynomial_trend': has_polynomial_trend,
            'polynomial_degree': poly_degree,
            'has_oscillations': has_oscillations,
            'dominant_frequencies': dominant_frequencies,
            'noise_level': noise_level,
            'is_smooth': is_smooth,
            'has_discontinuities': has_discontinuities,
            'boundary_importance': boundary_importance
        }

    @staticmethod
    def _estimate_polynomial_degree(t: np.ndarray, y: np.ndarray) -> float:
        """Estimate effective polynomial degree"""
        n = len(y)
        max_degree = min(7, n // 10)

        # Fit polynomials of increasing order
        residuals = []
        for degree in range(1, max_degree + 1):
            coeffs = np.polyfit(t, y, degree)
            fit = np.polyval(coeffs, t)
            residual = np.mean((y - fit)**2)
            residuals.append(residual)

        # Find elbow in residual curve
        if len(residuals) > 2:
            improvements = -np.diff(residuals)
            # Degree where improvement drops below threshold
            threshold = 0.1 * improvements[0]
            significant_degrees = np.where(improvements > threshold)[0]
            if len(significant_degrees) > 0:
                return float(significant_degrees[-1] + 2)

        return 1.0


class PyNumDiffMethodSelector:
    """
    Select optimal PyNumDiff method based on signal analysis
    Based on comprehensive testing with x^(5/2) + sin(2x) and other functions
    """

    # Performance data from our testing (RMSE values)
    METHOD_PERFORMANCE = {
        'butterdiff': {'rmse': 0.029, 'category': 'excellent'},
        'tv_velocity': {'rmse': 0.038, 'category': 'excellent'},
        'tvrdiff': {'rmse': 0.038, 'category': 'excellent'},
        'polydiff': {'rmse': 0.045, 'category': 'good'},
        'second_order': {'rmse': 0.074, 'category': 'good'},
        'splinediff': {'rmse': 0.092, 'category': 'good'},
        'mediandiff': {'rmse': 0.141, 'category': 'ok'},
        'fourth_order': {'rmse': 0.195, 'category': 'ok'},
        'first_order': {'rmse': 0.279, 'category': 'baseline'},
        # Methods to avoid
        'rbfdiff': {'rmse': 719.0, 'category': 'avoid', 'reason': 'conditioning_issues'},
        'kalman': {'rmse': 4.24, 'category': 'avoid', 'reason': 'model_mismatch'},
        'spectral': {'rmse': 1.75, 'category': 'avoid', 'reason': 'needs_periodic'},
    }

    @staticmethod
    def select_method(y: np.ndarray, dt: float,
                      derivative_order: int = 1,
                      priority: str = 'accuracy') -> Tuple[str, Dict]:
        """
        Select optimal PyNumDiff method based on signal characteristics

        Args:
            y: Signal to differentiate
            dt: Time step
            derivative_order: Order of derivative needed (1, 2, etc.)
            priority: 'accuracy', 'speed', or 'robustness'

        Returns:
            (method_name, parameters_dict)
        """
        # Analyze signal
        props = SignalAnalyzer.analyze(y, dt)

        # Decision tree based on our testing
        if props['has_polynomial_trend'] and props['has_oscillations']:
            # Mixed signals - TV methods excel here
            if props['boundary_importance'] > 0.3:
                # TV has best boundary handling
                method = 'tv_velocity'
                params = {'gamma': 1e-3}  # From our testing
            else:
                method = 'tvrdiff'
                params = {'order': derivative_order, 'gamma': 5e-3}

        elif props['is_smooth'] and not props['has_discontinuities']:
            # Smooth signals - Butterworth is best
            if props['noise_level'] < 0.01:
                method = 'butterdiff'
                cutoff = 0.2 if props['has_oscillations'] else 0.1
                params = {'filter_order': 2, 'cutoff_freq': cutoff}
            else:
                # More noise - use TV for robustness
                method = 'tv_velocity'
                params = {'gamma': props['noise_level'] * 0.01}

        elif props['has_polynomial_trend'] and props['polynomial_degree'] > 2:
            # High-order polynomial - use polynomial fitting
            method = 'polydiff'
            window = min(11, len(y) // 10)
            if window % 2 == 0:
                window += 1
            params = {
                'degree': min(int(props['polynomial_degree']) + 2, 7),
                'window_size': window,
                'kernel': 'friedrichs'
            }

        elif props['has_discontinuities']:
            # Discontinuous - use TV (handles jumps well)
            method = 'tvrdiff'
            params = {'order': derivative_order, 'gamma': 1e-2}

        elif priority == 'speed':
            # Need speed - use simple finite differences
            if derivative_order == 1:
                method = 'second_order'
                params = {}
            else:
                method = 'fourth_order'
                params = {}

        else:
            # Default fallback - spline is reliable
            method = 'splinediff'
            params = {'s': props['noise_level'] * 1e-3, 'degree': 3}

        # Add derivative order if > 1
        if derivative_order > 1 and method in ['tv_velocity', 'butterdiff', 'splinediff']:
            print(f"Warning: {method} optimized for 1st derivative. Consider alternative for order {derivative_order}")

        return method, params

    @staticmethod
    def get_method_info(method_name: str) -> Dict:
        """Get performance information about a method"""
        if method_name in PyNumDiffMethodSelector.METHOD_PERFORMANCE:
            return PyNumDiffMethodSelector.METHOD_PERFORMANCE[method_name]
        return {'category': 'unknown'}

    @staticmethod
    def recommend_methods(y: np.ndarray, dt: float, top_n: int = 3) -> list:
        """
        Recommend top N methods for a signal

        Returns list of (method_name, params, expected_performance)
        """
        props = SignalAnalyzer.analyze(y, dt)
        recommendations = []

        # Always try TV and Butterworth (our top performers)
        recommendations.append(('tv_velocity', {'gamma': 1e-3}, 'excellent'))
        recommendations.append(('butterdiff', {'filter_order': 2, 'cutoff_freq': 0.15}, 'excellent'))

        # Add specific recommendations based on signal
        if props['has_polynomial_trend']:
            window = min(11, len(y) // 10)
            if window % 2 == 0:
                window += 1
            recommendations.append((
                'polydiff',
                {'degree': 5, 'window_size': window, 'kernel': 'friedrichs'},
                'good'
            ))

        if props['is_smooth']:
            recommendations.append((
                'splinediff',
                {'s': 1e-4, 'degree': 3},
                'good'
            ))

        # Simple fallback
        recommendations.append(('second_order', {}, 'baseline'))

        return recommendations[:top_n]


def demo_method_selection():
    """Demonstrate method selection on test signals"""
    import matplotlib.pyplot as plt

    # Test signals
    t = np.linspace(0.1, 3, 301)
    dt = t[1] - t[0]

    test_signals = {
        'Polynomial + Oscillation': t**(5/2) + np.sin(2*t),
        'Pure Polynomial': t**3 - 2*t**2 + t,
        'Pure Oscillation': np.sin(2*np.pi*t) + 0.5*np.cos(6*np.pi*t),
        'Step Function': np.where(t < 1.5, 0, 1) + 0.1*np.sin(10*t),
        'Noisy Smooth': np.exp(-t) * np.cos(4*t) + 0.01*np.random.randn(len(t))
    }

    print("=" * 80)
    print("METHOD SELECTION DEMONSTRATION")
    print("=" * 80)

    for name, signal in test_signals.items():
        print(f"\n{name}:")
        print("-" * 40)

        # Analyze signal
        props = SignalAnalyzer.analyze(signal, dt)
        print(f"  Properties:")
        print(f"    - Polynomial trend: {props['has_polynomial_trend']}")
        print(f"    - Oscillations: {props['has_oscillations']}")
        print(f"    - Smooth: {props['is_smooth']}")
        print(f"    - Noise level: {props['noise_level']:.3f}")

        # Get recommendations
        method, params = PyNumDiffMethodSelector.select_method(signal, dt)
        print(f"\n  Selected method: {method}")
        print(f"  Parameters: {params}")

        info = PyNumDiffMethodSelector.get_method_info(method)
        print(f"  Expected performance: {info['category']}")

        # Get alternatives
        print(f"\n  Alternative methods:")
        for alt_method, alt_params, performance in PyNumDiffMethodSelector.recommend_methods(signal, dt, 3):
            if alt_method != method:
                print(f"    - {alt_method}: {performance}")


if __name__ == "__main__":
    demo_method_selection()