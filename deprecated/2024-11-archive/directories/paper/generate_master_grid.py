import pandas as pd
import numpy as np
import os

# --- Configuration ---
SUMMARY_FILE = 'build/results/comprehensive/comprehensive_summary.csv'
OUTPUT_DIR = 'gemini-analysis/paper/sections'
OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, 'appendix_c_master_grid.md')
EXCLUDE_METHODS = ['AAA-', 'Central-FD', 'TVRegDiff-Julia']

def categorize_noise(noise_level):
    if noise_level < 1e-5: return 'Near-Noiseless (<1e-5)'
    if noise_level <= 1e-3: return 'Low (1e-4 to 1e-3)'
    if noise_level <= 2e-2: return 'High (1e-2 to 2e-2)'
    return 'Very High (5e-2)'

def categorize_order(order):
    if order <= 2: return 'Low Orders (0-2)'
    if order <= 5: return 'Mid Orders (3-5)'
    return 'High Orders (6-7)'

def generate_master_grid():
    """
    Generates a detailed grid comparing method ranks across noise and order regimes.
    """
    try:
        df = pd.read_csv(SUMMARY_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find summary file at {SUMMARY_FILE}")
        return

    # --- Data Filtering and Categorization ---
    for pattern in EXCLUDE_METHODS:
        df = df[~df['method'].str.contains(pattern, regex=False)]
    
    df['mean_nrmse'] = df['mean_nrmse'].apply(lambda x: min(x, 1000))

    df['noise_regime'] = df['noise_level'].apply(categorize_noise)
    df['order_regime'] = df['deriv_order'].apply(categorize_order)

    # --- Analysis ---
    # Calculate mean performance for each method within each combination of system, noise, and order regime
    perf = df.groupby(['ode_system', 'noise_regime', 'order_regime', 'method'])['mean_nrmse'].mean().reset_index()
    
    # Rank methods within each group
    perf['rank'] = perf.groupby(['ode_system', 'noise_regime', 'order_regime'])['mean_nrmse'].rank()
    
    # Average the ranks across the ODE systems
    avg_ranks = perf.groupby(['noise_regime', 'order_regime', 'method'])['rank'].mean().reset_index()

    # --- Pivot to create the grid ---
    grid = avg_ranks.pivot_table(
        index='method',
        columns=['noise_regime', 'order_regime'],
        values='rank'
    )

    # --- Formatting ---
    # Define a logical column order for the final table
    noise_order = ['Near-Noiseless (<1e-5)', 'Low (1e-4 to 1e-3)', 'High (1e-2 to 2e-2)', 'Very High (5e-2)']
    order_order = ['Low Orders (0-2)', 'Mid Orders (3-5)', 'High Orders (6-7)']
    
    # Reorder the columns
    grid = grid.reindex(columns=pd.MultiIndex.from_product([noise_order, order_order]))
    
    # Sort methods by their performance in the most challenging bucket: Very High Noise, High Orders
    sorter = grid.loc[:, ('Very High (5e-2)', 'High Orders (6-7)')].sort_values(ascending=True).index
    grid = grid.reindex(sorter)

    # Format the table for Markdown
    grid_formatted = grid.applymap(lambda x: f'{x:.1f}' if pd.notna(x) else '---')
    
    # --- Save and Print ---
    markdown_output = "## Appendix C: Master Grid of Method Ranks\n\n"
    markdown_output += "This table shows the average rank of each contending method across three ODE systems. Ranks are calculated within each bucket defined by a Noise Regime and a Derivative Order Regime. Lower ranks are better. `---` indicates the method does not support that derivative order range.\n\n"
    markdown_output += grid_formatted.to_markdown()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILENAME, 'w') as f:
        f.write(markdown_output)
        
    print(f"Master grid table saved to {OUTPUT_FILENAME}")
    print("\n--- MASTER GRID (Excerpt) ---")
    print(grid_formatted.head())


if __name__ == '__main__':
    generate_master_grid()
