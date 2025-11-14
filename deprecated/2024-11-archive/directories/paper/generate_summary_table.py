import pandas as pd
import numpy as np
import os

# --- Configuration ---
SUMMARY_FILE = 'build/results/comprehensive/comprehensive_summary.csv'
OUTPUT_DIR = 'gemini-analysis/paper/sections'
OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, 'master_summary_table.md')
EXCLUDE_METHODS = ['AAA-', 'Central-FD', 'TVRegDiff-Julia']

def generate_master_table():
    """
    Generates the master summary table by analyzing method performance across different
    ODE systems and noise regimes, replicating the logic from the project's own scripts.
    """
    try:
        df = pd.read_csv(SUMMARY_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find summary file at {SUMMARY_FILE}")
        return

    # --- Data Filtering ---
    # 1. Exclude methods as per our narrative plan
    for pattern in EXCLUDE_METHODS:
        df = df[~df['method'].str.contains(pattern, regex=False)]
    
    # 2. Handle catastrophic failures by capping nRMSE values, replicating project script logic
    df['mean_nrmse'] = df['mean_nrmse'].apply(lambda x: min(x, 1000))

    # --- Define Noise Regimes ---
    df_low_noise = df[df['noise_level'] <= 0.001].copy()
    df_high_noise = df[df['noise_level'] >= 0.01].copy()

    def get_ranked_df(df_regime, regime_name):
        # First, calculate the mean performance of each method within each ODE system
        perf_per_system = df_regime.groupby(['ode_system', 'method'])['mean_nrmse'].mean().reset_index()
        # Then, rank methods within each system
        perf_per_system['rank'] = perf_per_system.groupby('ode_system')['mean_nrmse'].rank()
        # Finally, average the ranks and nRMSE across systems for each method
        agg_df = perf_per_system.groupby('method').agg(
            avg_rank=(f'rank', 'mean'),
            avg_nrmse=(f'mean_nrmse', 'mean')
        ).reset_index()
        agg_df.columns = ['Method', f'Avg. Rank ({regime_name})', f'Avg. nRMSE ({regime_name})']
        return agg_df

    # --- Process Each Regime ---
    low_noise_ranks = get_ranked_df(df_low_noise, 'Low Noise')
    high_noise_ranks = get_ranked_df(df_high_noise, 'High Noise')

    # --- Merge and Format ---
    df_merged = pd.merge(low_noise_ranks, high_noise_ranks, on='Method')
    df_merged = df_merged.sort_values(by='Avg. Rank (High Noise)', ascending=True)

    # Format for Markdown
    for col in ['Avg. Rank (Low Noise)', 'Avg. Rank (High Noise)']:
        df_merged[col] = df_merged[col].map('{:.1f}'.format)
    for col in ['Avg. nRMSE (Low Noise)', 'Avg. nRMSE (High Noise)']:
        df_merged[col] = df_merged[col].map('{:.3f}'.format)

    # --- Save and Print Table ---
    markdown_table = df_merged.to_markdown(index=False)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILENAME, 'w') as f:
        f.write(markdown_table)
        
    print(f"Master summary table saved to {OUTPUT_FILENAME}")
    print("\n--- ACCURATE MASTER SUMMARY TABLE ---")
    print(markdown_table)

if __name__ == '__main__':
    generate_master_table()
