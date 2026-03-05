#!/usr/bin/env python3
"""
Re-evaluate FL models on LOCAL data (per-institution) to match Paper Table 3
Combines with corrected Local + Central results
Saves to new folder: results/2006_10epoch__100rounds_final
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

sys.path.insert(0, 'src')

from models import CreditLSTMPaper
from calibration import compute_ece, compute_brier_score, get_calibrator
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

# Configuration
DATA_FILE = '../data/replication_dataset_finals.csv'
OLD_FL_DIR = "results/2006_10epoch__100rounds"  # Where FL models are
CORRECTED_DIR = "results/2006_10epoch__100rounds_corrected"  # Where Local/Central are
OUTPUT_DIR = "results/2006_10epoch__100rounds_final"  # NEW output
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128
MAX_SEQ_LEN = 60

print(f"💻 Device: {DEVICE}")
print(f"📂 Output: {OUTPUT_DIR}")

# Dataset class (same as before)
class FixedEncodingDataset(Dataset):
    def __init__(self, df, feature_columns, scaler=None, max_seq_len=60):
        self.df = df.copy()
        self.max_seq_len = max_seq_len
        self.feature_columns = feature_columns
        
        for col in feature_columns:
            if col not in self.df.columns:
                self.df[col] = 0
        
        self.data = self.df[feature_columns].fillna(0).values
        if scaler is not None:
            self.data = scaler.transform(self.data)
        
        self.labels = self.df['DEFAULT_LABEL'].values.astype(np.float32)
        self.loans = self.df['LOAN_SEQUENCE_NUMBER'].values
        
        self.loan_to_indices = {}
        for idx, loan in enumerate(self.loans):
            if loan not in self.loan_to_indices:
                self.loan_to_indices[loan] = []
            self.loan_to_indices[loan].append(idx)
        
        self.unique_loans = list(self.loan_to_indices.keys())
    
    def __len__(self):
        return len(self.unique_loans)
    
    def __getitem__(self, idx):
        loan_id = self.unique_loans[idx]
        indices = self.loan_to_indices[loan_id]
        
        seq = self.data[indices]
        label = self.labels[indices[-1]]
        
        if len(seq) > self.max_seq_len:
            seq = seq[-self.max_seq_len:]
        
        seq_len = len(seq)
        return torch.FloatTensor(seq), torch.FloatTensor([label]), seq_len

def collate_fn(batch):
    sequences, labels, lengths = zip(*batch)
    lengths = torch.LongTensor(lengths)
    sorted_idx = torch.argsort(lengths, descending=True)
    
    sequences = [sequences[i] for i in sorted_idx]
    labels = torch.cat([labels[i] for i in sorted_idx])
    lengths = lengths[sorted_idx]
    
    padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded, labels, lengths

def get_model_predictions(model, dataloader, device):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels, lengths in dataloader:
            if len(sequences) == 0:
                continue
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences, lengths)
            probs = outputs.cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy().flatten())
    return np.array(all_probs), np.array(all_labels)

def compute_metrics(probs, labels):
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

print("\n" + "="*80)
print("RE-EVALUATING FL MODELS ON LOCAL DATA (Per-Institution)")
print("Matching Paper Table 3 Specification")
print("="*80 + "\n")

# Load data and prepare same as training
print("[1/5] Loading and preparing data...")
df = pd.read_csv(DATA_FILE)
df['year'] = df['MONTHLY_REPORTING_PERIOD'].astype(str).str[:4].astype(int)

train_df = df[df['year'].isin([2006, 2007])].copy()
val_df = df[df['year'] == 2008].copy()
test_df = df[df['year'] == 2009].copy()

# Get top 14 institutions
seller_counts = train_df.groupby('SELLER_NAME')['LOAN_SEQUENCE_NUMBER'].nunique().sort_values(ascending=False)
TOP_14 = seller_counts.head(14).index.tolist()

# Filter to top 14
train_df = train_df[train_df['SELLER_NAME'].isin(TOP_14)]
val_df = val_df[val_df['SELLER_NAME'].isin(TOP_14)]
test_df = test_df[test_df['SELLER_NAME'].isin(TOP_14)]

# Prepare features
exclude = ['LOAN_SEQUENCE_NUMBER', 'MONTHLY_REPORTING_PERIOD', 'DEFAULT_LABEL', 
           'ZERO_BALANCE_CODE', 'year', 'month', 'SELLER_NAME']

train_encoded = pd.get_dummies(train_df, columns=train_df.select_dtypes(include=['object']).columns.tolist(), dummy_na=True)
val_encoded = pd.get_dummies(val_df, columns=val_df.select_dtypes(include=['object']).columns.tolist(), dummy_na=True)
test_encoded = pd.get_dummies(test_df, columns=test_df.select_dtypes(include=['object']).columns.tolist(), dummy_na=True)

feature_cols = [c for c in train_encoded.columns if c not in exclude]
scaler = StandardScaler()
scaler.fit(train_encoded[feature_cols].fillna(0))

input_dim = len(feature_cols)
print(f"  Features: {input_dim}")
print(f"  Top 14 institutions loaded")

# Copy Local and Central results from corrected folder
print("\n[2/5] Copying Local and Central results from corrected folder...")
import shutil
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Copy all Local-* folders
for local_folder in Path(CORRECTED_DIR).glob("Local-*"):
    dst = Path(OUTPUT_DIR) / local_folder.name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(local_folder, dst)
    print(f"  Copied {local_folder.name}")

# Copy Central
central_src = Path(CORRECTED_DIR) / "Central"
central_dst = Path(OUTPUT_DIR) / "Central"
if central_dst.exists():
    shutil.rmtree(central_dst)
shutil.copytree(central_src, central_dst)
print(f"  Copied Central")

print("\n[3/5] Re-evaluating FL models on LOCAL data (per-institution)...")

# Function to load FL model and test on each institution
def evaluate_fl_on_local(fl_model_path, scenario_name, active_sellers):
    """Test FL model on each institution's local test data separately"""
    
    # Load FL model
    model = CreditLSTMPaper(input_dim=input_dim).to(DEVICE)
    # Note: We can't actually load the model since we don't have saved weights
    # Instead, we'll need to load predictions from existing CSV files
    # and just reorganize them by institution
    
    print(f"  {scenario_name}: Re-organizing predictions by institution...")
    
    # Load existing FL predictions
    fl_pred_path = Path(OLD_FL_DIR) / scenario_name.replace(' ', '_').replace('(', '').replace(')', '')
    
    all_institution_results = {}
    
    for method in ['uncalibrated', 'platt', 'isotonic', 'temperature', 'beta']:
        if method == 'uncalibrated':
            csv_file = fl_pred_path / "uncalibrated_predictions.csv"
        else:
            csv_file = fl_pred_path / f"{method}_calibrated_predictions.csv"
        
        if not csv_file.exists():
            continue
        
        # Load FL predictions
        fl_preds = pd.read_csv(csv_file)
        
        # We need to map these back to institutions
        # Since we don't have institution labels in the prediction file,
        # we'll aggregate all predictions (current behavior is correct for pooled testing)
        
        all_institution_results[method] = compute_metrics(
            fl_preds['PREDICTED_PROB'].values,
            fl_preds['TRUE_LABEL'].values
        )
    
    return all_institution_results

# Actually, we realize we CAN'T properly re-evaluate without the model weights
# The predictions in CSV don't have institution labels
# So we'll note this limitation

print("  ⚠️ Limitation: Cannot re-evaluate without saved model weights")
print("  ⚠️ Copying FL folders as-is (aggregated testing)")
print("  ⚠️ TRUE per-institution testing would require saved FL models")

# Copy FL folders
for fl_scenario in ['FL_n', 'FL_n-1', 'FL_n-2']:
    src = Path(OLD_FL_DIR) / fl_scenario
    dst = Path(OUTPUT_DIR) / fl_scenario
    if src.exists():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"  Copied {fl_scenario}")

print("\n[4/5] Generating combined results JSON...")

# Load all results and create JSON
from generate_visualizations import compute_metrics as compute_from_csv

all_results = {}

# Load Local (aggregate all 14)
print("  Loading Local results...")
local_folders = list(Path(OUTPUT_DIR).glob("Local-*"))
for method in ['uncalibrated', 'platt', 'isotonic', 'temperature', 'beta']:
    all_probs = []
    all_labels = []
    for folder in local_folders:
        csv_file = folder / ("uncalibrated_predictions.csv" if method == 'uncalibrated' else f"{method}_calibrated_predictions.csv")
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            all_probs.extend(df['PREDICTED_PROB'].values)
            all_labels.extend(df['TRUE_LABEL'].values)
    if all_probs:
        all_results.setdefault('Local', {})[method] = compute_metrics(np.array(all_probs), np.array(all_labels))

# Load Central
print("  Loading Central results...")
for method in ['uncalibrated', 'platt', 'isotonic', 'temperature', 'beta']:
    csv_file = Path(OUTPUT_DIR) / "Central" / ("uncalibrated_predictions.csv" if method == 'uncalibrated' else f"{method}_calibrated_predictions.csv")
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        all_results.setdefault('Central', {})[method] = compute_metrics(df['PREDICTED_PROB'].values, df['TRUE_LABEL'].values)

# Load FL scenarios
for fl_name in ['FL (n)', 'FL (n-1)', 'FL (n-2)']:
    print(f"  Loading {fl_name} results...")
    folder = Path(OUTPUT_DIR) / fl_name.replace(' ', '_').replace('(', '').replace(')', '')
    for method in ['uncalibrated', 'platt', 'isotonic', 'temperature', 'beta']:
        csv_file = folder / ("uncalibrated_predictions.csv" if method == 'uncalibrated' else f"{method}_calibrated_predictions.csv")
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            all_results.setdefault(fl_name, {})[method] = compute_metrics(df['PREDICTED_PROB'].values, df['TRUE_LABEL'].values)

# Save JSON
calibration_dir = Path(OUTPUT_DIR) / "calibration"
calibration_dir.mkdir(exist_ok=True)

metrics_only = {}
for scenario, methods in all_results.items():
    metrics_only[scenario] = {}
    for method, results in methods.items():
        metrics_only[scenario][method] = {k: float(v) for k, v in results.items() if k not in ['probs', 'labels']}

with open(calibration_dir / 'calibration_results_all.json', 'w') as f:
    json.dump(metrics_only, f, indent=2)

print(f"\n  ✅ Saved: {calibration_dir}/calibration_results_all.json")

print("\n[5/5] Complete!")
print("="*80)
print(f"\n📂 Results saved to: {OUTPUT_DIR}/")
print("\nCurrent status:")
print("  ✅ Local: 14 models, tested on own data (CORRECT)")
print("  ✅ Central: Tested on pooled data (CORRECT)")  
print("  ⚠️ FL: Tested on aggregated data (NOT per-institution)")
print("\nLimitation: True per-institution FL testing requires saved model weights")
print("="*80)
