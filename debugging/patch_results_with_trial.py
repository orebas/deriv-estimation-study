#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sys

ROOT = Path('/home/orebas/tmp/deriv-estimation-study')
RES_DIR = ROOT / 'build' / 'results' / 'comprehensive'
INPUT_DIR = ROOT / 'build' / 'data' / 'input'
OUTPUT_DIR = ROOT / 'build' / 'data' / 'output'
PRED_DIR = RES_DIR / 'predictions'

METHOD = 'GP_RBF_Python'
CATEGORY = 'Gaussian Process'
LANG = 'Python'


def compute_metrics(indata: dict, outdata: dict):
    ode_key = indata['ode_key']
    noise_level = float(indata['config']['noise_level'])
    trial = int(indata['config']['trial'])
    times = np.array(indata['times'], dtype=float)
    gt = {int(k): np.array(v, dtype=float) for k, v in indata['ground_truth_derivatives'].items()}

    method_res = outdata['methods'][METHOD]
    timing = float(method_res.get('timing', 0.0))
    preds = {int(k): np.array(v, dtype=float) for k, v in method_res['predictions'].items()}

    rows = []
    for order, pred in preds.items():
        true_vals = gt[order]
        finite = np.isfinite(pred)
        if pred.size < 3:
            continue
        idx = np.arange(pred.size)
        mask = finite & (idx > 0) & (idx < pred.size - 1)
        if not np.any(mask):
            continue
        rmse = float(np.sqrt(np.mean((pred[mask] - true_vals[mask]) ** 2)))
        mae = float(np.mean(np.abs(pred[mask] - true_vals[mask])))
        std_true = float(np.std(true_vals[mask]))
        nrmse = float(rmse / max(std_true, 1e-12))
        rows.append({
            'ode_system': ode_key,
            'noise_level': noise_level,
            'trial': trial,
            'method': METHOD,
            'category': CATEGORY,
            'language': LANG,
            'deriv_order': order,
            'rmse': rmse,
            'mae': mae,
            'nrmse': nrmse,
            'timing': timing,
            'valid_points': int(np.sum(mask)),
            'total_points': int(pred.size - 2),
        })
    return rows


def patch_predictions_json(trial_id: str, outdata: dict):
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = PRED_DIR / f'{trial_id}.json'
    if not pred_path.exists():
        return
    with pred_path.open('r') as f:
        combined = json.load(f)
    method_res = outdata['methods'][METHOD]
    # Overwrite method entry
    combined['methods'][METHOD] = {
        'predictions': {str(k): [float(x) for x in v] for k, v in method_res['predictions'].items()},
        'timing': float(method_res.get('timing', 0.0)),
        'success': bool(method_res.get('success', True)),
        'language': 'Python',
    }
    with pred_path.open('w') as f:
        json.dump(combined, f)


def regenerate_summary(results_csv: Path, summary_csv: Path):
    df = pd.read_csv(results_csv)
    grp_cols = ['ode_system', 'method', 'category', 'language', 'deriv_order', 'noise_level']
    agg = df.groupby(grp_cols).agg(
        mean_rmse=('rmse', 'mean'),
        std_rmse=('rmse', 'std'),
        min_rmse=('rmse', 'min'),
        max_rmse=('rmse', 'max'),
        mean_mae=('mae', 'mean'),
        mean_nrmse=('nrmse', 'mean'),
        std_nrmse=('nrmse', 'std'),
        min_nrmse=('nrmse', 'min'),
        max_nrmse=('nrmse', 'max'),
        mean_timing=('timing', 'mean'),
        trials=('rmse', 'count'),
    ).reset_index()
    agg.to_csv(summary_csv, index=False)


def main(trial_id: str):
    in_path = INPUT_DIR / f'{trial_id}.json'
    out_path = OUTPUT_DIR / f'{trial_id}_results.json'
    results_csv = RES_DIR / 'comprehensive_results.csv'
    summary_csv = RES_DIR / 'comprehensive_summary.csv'

    if not in_path.exists() or not out_path.exists():
        print('Missing input/output JSON:', in_path, out_path)
        sys.exit(1)

    with in_path.open('r') as f:
        indata = json.load(f)
    with out_path.open('r') as f:
        outdata = json.load(f)

    # Compute new rows for all orders
    rows = compute_metrics(indata, outdata)
    if not rows:
        print('No rows computed; aborting')
        sys.exit(2)

    # Load results CSV and drop old rows for this trial/method
    df = pd.read_csv(results_csv)
    ode_key = indata['ode_key']
    noise_level = float(indata['config']['noise_level'])
    trial = int(indata['config']['trial'])

    mask = (
        (df['ode_system'] == ode_key) &
        (df['noise_level'] == noise_level) &
        (df['trial'] == trial) &
        (df['method'] == METHOD) &
        (df['language'] == LANG)
    )
    before = len(df)
    df = df.loc[~mask].copy()

    # Append new rows in the same column order
    cols = ['ode_system','noise_level','trial','method','category','language','deriv_order','rmse','mae','nrmse','timing','valid_points','total_points']
    add_df = pd.DataFrame(rows)[cols]
    df = pd.concat([df, add_df], ignore_index=True)
    df.to_csv(results_csv, index=False)
    print(f'Replaced {before - len(df)} rows and appended {len(add_df)} rows for {trial_id}:{METHOD}')

    # Update combined predictions JSON (best-effort)
    try:
        patch_predictions_json(trial_id, outdata)
    except Exception as e:
        print('Warning: could not patch predictions JSON:', e)

    # Recompute summary
    regenerate_summary(results_csv, summary_csv)
    print('Updated summary at', summary_csv)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: patch_results_with_trial.py <trial_id>')
        sys.exit(1)
    main(sys.argv[1])
