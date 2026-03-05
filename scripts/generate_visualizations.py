#!/usr/bin/env python3
"""
Generate visualizations from experiment results.
Reads predictions from results/experiments/ and saves figures to results/figures/.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, 'src')

from visualization import create_all_visualizations
from calibration import compute_ece, compute_brier_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Directories — must match what src/train.py produces
base_dir = Path("results/experiments")
save_dir = Path("results/figures")
save_dir.mkdir(parents=True, exist_ok=True)

print(f"Loading results from: {base_dir}")
print("="*80)

# Load all predictions from CSV files
all_results = {}

scenarios = ['Local', 'Central', 'FL_n', 'FL_n-1', 'FL_n-2']
scenario_names_display = {
    'Local': 'Local',
    'Central': 'Central',
    'FL_n': 'FL (n)',
    'FL_n-1': 'FL (n-1)',
    'FL_n-2': 'FL (n-2)'
}

def compute_metrics(probs, labels):
    """Compute all metrics from predictions"""
    preds = (probs >= 0.5).astype(int)
    return {
        'probs': probs,
        'labels': labels,
        'ece': compute_ece(probs, labels),
        'brier': compute_brier_score(probs, labels),
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, zero_division=0),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0)
    }

for scenario in scenarios:
    scenario_display = scenario_names_display[scenario]

    # For Local, aggregate across all institution folders
    if scenario == 'Local':
        print(f"\nLoading {scenario} scenario (aggregating all local models)...")
        local_folders = list(base_dir.glob("Local/*"))
        print(f"  Found {len(local_folders)} local model folders")

        all_results[scenario_display] = {}

        for method in ['uncalibrated', 'platt', 'isotonic', 'temperature', 'beta']:
            all_probs = []
            all_labels = []

            for local_folder in local_folders:
                csv_file = local_folder / "predictions.csv"
                if csv_file.exists():
                    df = pd.read_csv(csv_file)
                    col = 'prob_uncalibrated' if method == 'uncalibrated' else f'prob_{method}'
                    if col in df.columns:
                        all_probs.extend(df[col].values)
                        all_labels.extend(df['true_label'].values)

            if all_probs:
                all_results[scenario_display][method] = compute_metrics(
                    np.array(all_probs), np.array(all_labels)
                )
                print(f"    {method}: {len(all_probs)} predictions")
    else:
        # For FL and Central scenarios
        scenario_dir = base_dir / scenario
        if not scenario_dir.exists():
            print(f"\n⚠️  Skipping {scenario_display} — folder not found at {scenario_dir}")
            continue

        print(f"\nLoading {scenario_display} scenario...")
        all_results[scenario_display] = {}

        # Aggregate across institution sub-folders
        inst_folders = [f for f in scenario_dir.iterdir() if f.is_dir() and f.name != 'calibrators']
        if not inst_folders:
            inst_folders = [scenario_dir]

        for method in ['uncalibrated', 'platt', 'isotonic', 'temperature', 'beta']:
            all_probs = []
            all_labels = []
            for folder in inst_folders:
                csv_file = folder / "predictions.csv"
                if csv_file.exists():
                    df = pd.read_csv(csv_file)
                    col = 'prob_uncalibrated' if method == 'uncalibrated' else f'prob_{method}'
                    if col in df.columns:
                        all_probs.extend(df[col].values)
                        all_labels.extend(df['true_label'].values)

            if all_probs:
                all_results[scenario_display][method] = compute_metrics(
                    np.array(all_probs), np.array(all_labels)
                )
                print(f"  {method}: {len(all_probs)} predictions")
            else:
                print(f"  ⚠️  {method}: no predictions found")

print("\n" + "="*80)
print(f"Loaded {len(all_results)} scenarios")
print("\nGenerating 6 visualizations...")
print("="*80 + "\n")

create_all_visualizations(all_results, save_dir)

print("\n" + "="*80)
print(f"✅ All visualizations saved to: {save_dir}/")
print("="*80)
