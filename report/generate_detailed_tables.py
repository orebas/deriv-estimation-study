#!/usr/bin/env python3
"""
Generate detailed LaTeX tables for supplementary material.
Reads comprehensive_summary.csv and creates:
- Part I: Method × Noise Level tables (one per derivative order)
- Part II: Noise × Derivative Order tables (one per full-coverage method)
"""

import csv
import sys
from collections import defaultdict

# Read CSV data
data = defaultdict(dict)  # data[(method, order, noise)] = nrmse
methods_set = set()
full_coverage_methods = set()

csv_path = '../results/comprehensive/comprehensive_summary.csv'

with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        method = row['method']
        order = int(row['deriv_order'])
        noise = float(row['noise_level'])
        nrmse = float(row['mean_nrmse'])

        data[(method, order, noise)] = nrmse
        methods_set.add(method)

# Identify full-coverage methods
noise_levels = [1e-8, 1e-6, 1e-4, 1e-3, 0.01, 0.02, 0.05]
orders = list(range(8))

for method in methods_set:
    coverage = sum(1 for o in orders for n in noise_levels if (method, o, n) in data)
    if coverage == 56:
        full_coverage_methods.add(method)

# Sort methods alphabetically
all_methods_sorted = sorted(methods_set)
full_coverage_sorted = sorted(full_coverage_methods)

def format_nrmse(val):
    """Format nRMSE value for LaTeX table."""
    if val != val:  # NaN check
        return '---'
    if val < 0.001:
        return f'{val:.2e}'
    elif val < 1:
        return f'{val:.3f}'
    elif val < 100:
        return f'{val:.1f}'
    else:
        return f'{val:.1e}'

def escape_method_name(name):
    """Escape underscores in method names for LaTeX."""
    return name.replace('_', '\\_')

# Generate Part I: Method × Noise tables (one per order)
part1_tables = []

for order in orders:
    table_lines = []
    table_lines.append(f'\\subsection*{{Table S{order+1}: Derivative Order {order}}}')
    table_lines.append('')
    table_lines.append('\\begin{longtable}{l' + 'r' * len(noise_levels) + '}')
    table_lines.append('\\toprule')

    # Header
    header = '\\textbf{Method}'
    for noise in noise_levels:
        if noise < 0.001:
            header += f' & \\textbf{{{noise:.0e}}}'
        else:
            header += f' & \\textbf{{{noise:.3f}}}'
    header += ' \\\\'
    table_lines.append(header)
    table_lines.append('\\midrule')
    table_lines.append('\\endhead')

    # Data rows
    for method in all_methods_sorted:
        row = escape_method_name(method)
        for noise in noise_levels:
            key = (method, order, noise)
            if key in data:
                row += f' & {format_nrmse(data[key])}'
            else:
                row += ' & ---'
        row += ' \\\\'
        table_lines.append(row)

    table_lines.append('\\bottomrule')
    table_lines.append('\\end{longtable}')
    table_lines.append('')

    part1_tables.append('\n'.join(table_lines))

# Generate Part II: Noise × Derivative Order tables (one per method)
part2_tables = []

for idx, method in enumerate(full_coverage_sorted):
    table_lines = []
    table_lines.append(f'\\subsection*{{Table M{idx+1}: {escape_method_name(method)}}}')
    table_lines.append('')
    table_lines.append('\\begin{longtable}{l' + 'r' * len(orders) + '}')
    table_lines.append('\\toprule')

    # Header
    header = '\\textbf{Noise Level}'
    for order in orders:
        header += f' & \\textbf{{Ord {order}}}'
    header += ' \\\\'
    table_lines.append(header)
    table_lines.append('\\midrule')
    table_lines.append('\\endhead')

    # Data rows
    for noise in noise_levels:
        if noise < 0.001:
            row = f'{noise:.0e}'
        else:
            row = f'{noise:.3f}'

        for order in orders:
            key = (method, order, noise)
            row += f' & {format_nrmse(data[key])}'
        row += ' \\\\'
        table_lines.append(row)

    table_lines.append('\\bottomrule')
    table_lines.append('\\end{longtable}')
    table_lines.append('')

    part2_tables.append('\n'.join(table_lines))

# Output the tables
print("% PART I: METHOD × NOISE LEVEL TABLES")
print("% Generated automatically from comprehensive_summary.csv")
print()
for table in part1_tables:
    print(table)
    print('\\clearpage')
    print()

print()
print("% PART II: NOISE × DERIVATIVE ORDER TABLES")
print("% Generated automatically from comprehensive_summary.csv")
print()
for table in part2_tables:
    print(table)
    print('\\clearpage')
    print()

# Print summary statistics to stderr
print(f"Generated {len(part1_tables)} Part I tables (orders 0-7)", file=sys.stderr)
print(f"Generated {len(part2_tables)} Part II tables ({len(full_coverage_sorted)} full-coverage methods)", file=sys.stderr)
print(f"Total methods: {len(all_methods_sorted)}", file=sys.stderr)
print(f"Full-coverage methods: {len(full_coverage_sorted)}", file=sys.stderr)
