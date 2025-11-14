#!/usr/bin/env python3
"""
Prepare data for speed-accuracy plot without pandas.
"""
import csv
from collections import defaultdict

# Read and process data
data_by_method = defaultdict(lambda: {'times': [], 'errors': []})

with open('/tmp/filtered_data.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        method = row['method']
        nrmse = float(row['mean_nrmse'])
        timing = float(row['mean_timing'])
        data_by_method[method]['times'].append(timing)
        data_by_method[method]['errors'].append(nrmse)

# Calculate averages
results = []
for method, values in data_by_method.items():
    avg_time = sum(values['times']) / len(values['times'])
    avg_error = sum(values['errors']) / len(values['errors'])
    results.append((method, avg_time, avg_error))

# Sort by time
results.sort(key=lambda x: x[1])

# Print results for gnuplot or manual plotting
print("# Method,Time(s),nRMSE")
for method, time, error in results:
    print(f"{method},{time:.6f},{error:.6f}")

# Print key statistics
print("\n# Summary:")
print(f"# Fastest: {results[0][0]} ({results[0][1]:.3f}s)")
min_error = min(results, key=lambda x: x[2])
print(f"# Most accurate: {min_error[0]} (nRMSE={min_error[2]:.4f})")
print(f"# Speed range: {results[0][1]:.3f}s to {results[-1][1]:.1f}s ({results[-1][1]/results[0][1]:.0f}x)")

# Categorize methods
categories = {
    'GP': ['GP-TaylorAD-Julia', 'GP-RBF-Python', 'GP-RBF-Iso-Python', 'GP-RBF-MeanSub-Python',
           'GP-Julia-SE', 'GP-Matern-1.5-Julia', 'GP-Matern-2.5-Julia'],
    'Spectral': ['Fourier-GCV', 'Fourier-Basic-Python', 'Fourier-Continuation-Python', 'Fourier-Adaptive-Julia',
                 'Fourier-Adaptive-Python', 'Fourier-Continuation-Adaptive',
                 'Chebyshev-Basic-Python', 'Chebyshev-AICc', 'Fourier-Interp'],
    'Spline': ['Spline-Dierckx-5', 'Spline-GSS', 'Spline-Dierckx-3'],
    'Filter': ['Savitzky-Golay', 'SavitzkyGolay-Fixed', 'SavitzkyGolay-Adaptive',
               'SavitzkyGolay-Julia-Fixed', 'SavitzkyGolay-Julia-Adaptive', 'SavitzkyGolay-Julia-Hybrid']
}

print("\n# Methods on Pareto frontier (best error for given speed):")
best_error = float('inf')
pareto_methods = []
for method, time, error in results:
    if error < best_error:
        best_error = error
        # Find category
        category = 'Other'
        for cat, methods in categories.items():
            if method in methods:
                category = cat
                break
        pareto_methods.append((method, time, error, category))
        print(f"#   {method:30s} {time:8.3f}s  nRMSE={error:.4f}  ({category})")

# Save Pareto frontier for LaTeX
with open('../build/tables/publication/pareto_frontier.txt', 'w') as f:
    f.write("Method & Time(s) & nRMSE & Category\n")
    for method, time, error, cat in pareto_methods:
        f.write(f"{method} & {time:.3f} & {error:.4f} & {cat}\n")