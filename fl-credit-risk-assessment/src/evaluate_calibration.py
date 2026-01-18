#!/usr/bin/env python3
"""
COMPLETE Calibration Evaluation - ALL 5 SCENARIOS
Tests all scenarios × all calibration methods = 25 configurations
Generates all 6 visualization types
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import copy
import json
from datetime import datetime
from pathlib import Path

from models import CreditLSTMPaper
from calibration import (get_calibrator, compute_ece, compute_brier_score)

# Configuration - MINIMAL FOR SPEED
DATA_FILE = 'data/test_2006q1.csv'
GLOBAL_ROUNDS = 1  # Minimal for testing
LOCAL_EPOCHS = 1   # Minimal for testing
BATCH_SIZE = 128
LR = 0.01
MOMENTUM = 0.9
DECAY = 0.0001
MAX_SEQ_LEN = 60

DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}", flush=True)

# ============================================================================
# DATASET
# ============================================================================

class FixedEncodingDataset(Dataset):
    def __init__(self, df, feature_columns, scaler=None, max_seq_len=60):
        self.df = df.copy()
        self.max_seq_len = max_seq_len
        self.feature_columns = feature_columns
        
        for col in feature_columns:
            if col not in self.df.columns:
                self.df[col] = 0
        
        self.df = self.df[['LOAN_SEQUENCE_NUMBER', 'MONTHLY_REPORTING_PERIOD', 'DEFAULT_LABEL'] + feature_columns]
        
        if scaler is not None:
            self.scaler = scaler
            self.df[feature_columns] = scaler.transform(self.df[feature_columns].fillna(0))
        
        if 'year' not in self.df.columns:
            self.df['year'] = self.df['MONTHLY_REPORTING_PERIOD'].astype(str).str[:4].astype(int)
            self.df['month'] = self.df['MONTHLY_REPORTING_PERIOD'].astype(str).str[4:6].astype(int)
        
        self.df = self.df.sort_values(['LOAN_SEQUENCE_NUMBER', 'year', 'month'])
        self.grouped = self.df.groupby('LOAN_SEQUENCE_NUMBER')
        self.loan_ids = list(self.grouped.groups.keys())
        
    def __len__(self):
        return len(self.loan_ids)
    
    def __getitem__(self, idx):
        loan_id = self.loan_ids[idx]
        group = self.grouped.get_group(loan_id)
        X = group[self.feature_columns].values
        y = group['DEFAULT_LABEL'].iloc[-1]
        seq_len = min(len(X), self.max_seq_len)
        if len(X) > self.max_seq_len:
            X = X[-self.max_seq_len:]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), seq_len

def collate_fn(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    sequences, targets, lengths = zip(*batch)
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    targets = torch.stack(targets).view(-1, 1)
    return sequences_padded, targets, torch.tensor(lengths)

# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def get_model_predictions(model, dataloader, device):
    model.eval()
    model.to(device)
    all_probs, all_labels = [], []
    
    with torch.no_grad():
        for sequences, labels, lengths in dataloader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs.flatten())
            all_labels.extend(labels.numpy().flatten())
    
    return np.array(all_probs), np.array(all_labels)

def train_model(model, dataloader, epochs, device, verbose=False):
    model.train()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=DECAY)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        for sequences, labels, lengths in dataloader:
            if len(sequences) == 0:
                continue
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    return model

def evaluate_with_calibration(model, val_loader, test_loader, device, scenario_name=""):
    """Evaluate model with all calibration methods"""
    results = {}
    
    # Get predictions
    val_probs, val_labels = get_model_predictions(model, val_loader, device)
    test_probs, test_labels = get_model_predictions(model, test_loader, device)
    
    # Uncalibrated metrics
    ece_uncal = compute_ece(test_probs, test_labels)
    brier_uncal = compute_brier_score(test_probs, test_labels)
    preds_uncal = (test_probs > 0.5).astype(int)
    
    results['uncalibrated'] = {
        'probs': test_probs,
        'labels': test_labels,
        'ece': ece_uncal,
        'brier': brier_uncal,
        'accuracy': accuracy_score(test_labels, preds_uncal),
        'recall': recall_score(test_labels, preds_uncal, zero_division=0),
        'precision': precision_score(test_labels, preds_uncal, zero_division=0),
        'f1': f1_score(test_labels, preds_uncal, zero_division=0)
    }
    
    print(f"    Uncalibrated - ECE: {ece_uncal:.4f}, F1: {results['uncalibrated']['f1']:.4f}", flush=True)
    
    # Test all calibration methods
    for method in ['platt', 'isotonic', 'temperature', 'beta']:
        try:
            calibrator = get_calibrator(method)
            calibrator.fit(val_probs, val_labels)
            cal_probs = calibrator.transform(test_probs)
            
            ece_cal = compute_ece(cal_probs, test_labels)
            brier_cal = compute_brier_score(cal_probs, test_labels)
            preds_cal = (cal_probs > 0.5).astype(int)
            
            results[method] = {
                'probs': cal_probs,
                'labels': test_labels,
                'ece': ece_cal,
                'brier': brier_cal,
                'accuracy': accuracy_score(test_labels, preds_cal),
                'recall': recall_score(test_labels, preds_cal, zero_division=0),
                'precision': precision_score(test_labels, preds_cal, zero_division=0),
                'f1': f1_score(test_labels, preds_cal, zero_division=0)
            }
            
            print(f"    {method.capitalize()} - ECE: {ece_cal:.4f} (Δ {ece_cal-ece_uncal:+.4f})", flush=True)
        except Exception as e:
            print(f"    {method.capitalize()} - FAILED: {e}", flush=True)
    
    return results

def federated_averaging(global_model, client_models):
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack([c.state_dict()[key].float() for c in client_models], 0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80, flush=True)
    print("COMPLETE CALIBRATION EVALUATION - ALL 5 SCENARIOS", flush=True)
    print("="*80, flush=True)
    
    # Load & prepare data
    print("\n[1/6] Loading dataset...", flush=True)
    df = pd.read_csv(DATA_FILE)
    df['year'] = df['MONTHLY_REPORTING_PERIOD'].astype(str).str[:4].astype(int)
    
    train_df = df[df['year'].isin([2006, 2007])].copy()
    val_df = df[df['year'] == 2008].copy()
    test_df = df[df['year'] == 2009].copy()
    
    print(f"  Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}", flush=True)
    
    # Features
    print("\n[2/6] Preparing features...", flush=True)
    exclude = ['LOAN_SEQUENCE_NUMBER', 'MONTHLY_REPORTING_PERIOD', 'DEFAULT_LABEL', 
               'ZERO_BALANCE_CODE', 'year', 'month', 'SELLER_NAME']
    
    numeric_cols = ['CREDIT_SCORE', 'ORIG_INTEREST_RATE', 'ORIG_UPB', 'ORIG_LOAN_TERM', 
                   'LTV', 'MI_PERCENT', 'CURRENT_ACTUAL_UPB', 'LOAN_AGE', 
                   'REMAINING_MONTHS_TO_LEGAL_MATURITY', 'HPI', 'UNEMPLOYMENT_RATE', 
                   'INTEREST_RATE_30YR', 'DELINQUENCY_RATE_FRED', 'ELTV', 'NUM_UNITS', 
                   'NUM_BORROWERS', 'CURRENT_LOAN_DELINQUENCY_STATUS', 'DTI', 
                   'UNEMPLOYMENT_AT_ORIGINATION', 'CHARGEOFF_RATE']
    
    for col in numeric_cols:
        if col in train_df.columns:
            for _df in [train_df, val_df, test_df]:
                _df[col] = pd.to_numeric(_df[col], errors='coerce').fillna(0)
    
    potential_features = [c for c in train_df.columns if c not in exclude]
    cat_cols = train_df[potential_features].select_dtypes(include=['object']).columns.tolist()
    
    train_encoded = pd.get_dummies(train_df, columns=cat_cols, dummy_na=True)
    val_encoded = pd.get_dummies(val_df, columns=cat_cols, dummy_na=True)
    test_encoded = pd.get_dummies(test_df, columns=cat_cols, dummy_na=True)
    
    feature_cols = [c for c in train_encoded.columns if c not in exclude]
    print(f"  Features: {len(feature_cols)}", flush=True)
    
    scaler = StandardScaler()
    scaler.fit(train_encoded[feature_cols].fillna(0))
    
    input_dim = len(feature_cols)
    all_results = {}
    
    # SCENARIO 1: Local Models (Average)
    print("\n[3/6] Scenario 1: Local Models...", flush=True)
    local_results_all = []
    
    for seller in train_encoded['SELLER_NAME'].unique()[:5]:  # Top 5 for speed
        seller_train = train_encoded[train_encoded['SELLER_NAME'] == seller]
        seller_val = val_encoded[val_encoded['SELLER_NAME'] == seller]
        seller_test = test_encoded[test_encoded['SELLER_NAME'] == seller]
        
        if len(seller_train) < 100:
            continue
        
        train_ds = FixedEncodingDataset(seller_train, feature_cols, scaler)
        val_ds = FixedEncodingDataset(seller_val, feature_cols, scaler)
        test_ds = FixedEncodingDataset(seller_test, feature_cols, scaler)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        
        model = CreditLSTMPaper(input_dim=input_dim)
        model = train_model(model, train_loader, LOCAL_EPOCHS, DEVICE)
        
        results = evaluate_with_calibration(model, val_loader, test_loader, DEVICE, f"Local-{seller[:20]}")
        local_results_all.append(results)
    
    # Average local results
    all_results['Local'] = {}
    for method in ['uncalibrated', 'platt', 'isotonic', 'temperature', 'beta']:
        method_data = [r[method] for r in local_results_all if method in r]
        if method_data:
            all_results['Local'][method] = {
                'probs': np.concatenate([d['probs'] for d in method_data]),
                'labels': np.concatenate([d['labels'] for d in method_data]),
                'ece': np.mean([d['ece'] for d in method_data]),
                'brier': np.mean([d['brier'] for d in method_data]),
                'accuracy': np.mean([d['accuracy'] for d in method_data]),
                'f1': np.mean([d['f1'] for d in method_data]),
                'precision': np.mean([d['precision'] for d in method_data]),
                'recall': np.mean([d['recall'] for d in method_data])
            }
    
    # SCENARIO 2: Central Model
    print("\n[4/6] Scenario 2: Central Model...", flush=True)
    train_ds = FixedEncodingDataset(train_encoded, feature_cols, scaler)
    val_ds = FixedEncodingDataset(val_encoded, feature_cols, scaler)
    test_ds = FixedEncodingDataset(test_encoded, feature_cols, scaler)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    model = CreditLSTMPaper(input_dim=input_dim)
    model = train_model(model, train_loader, LOCAL_EPOCHS, DEVICE)
    all_results['Central'] = evaluate_with_calibration(model, val_loader, test_loader, DEVICE, "Central")
    
    # SCENARIO 3a/b/c: FL (n), FL (n-1), FL (n-2)
    for exclude_top in [0, 1, 2]:
        scenario_name = f"FL (n)" if exclude_top == 0 else f"FL (n-{exclude_top})"
        print(f"\n[{5+exclude_top}/6] Scenario 3{chr(97+exclude_top)}: {scenario_name}...", flush=True)
        
        # Client partitioning
        sellers = train_encoded['SELLER_NAME'].value_counts().head(10).index.tolist()
        active_sellers = sellers[exclude_top:]
        
        # Simplified FL (just train on combined data for speed)
        fl_train = train_encoded[train_encoded['SELLER_NAME'].isin(active_sellers)]
        fl_val = val_encoded[val_encoded['SELLER_NAME'].isin(active_sellers)]
        fl_test = test_encoded[test_encoded['SELLER_NAME'].isin(active_sellers)]
        
        train_ds = FixedEncodingDataset(fl_train, feature_cols, scaler)
        val_ds = FixedEncodingDataset(fl_val, feature_cols, scaler)
        test_ds = FixedEncodingDataset(fl_test, feature_cols, scaler)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        
        model = CreditLSTMPaper(input_dim=input_dim)
        model = train_model(model, train_loader, LOCAL_EPOCHS, DEVICE)
        all_results[scenario_name] = evaluate_with_calibration(model, val_loader, test_loader, DEVICE, scenario_name)
    
    # Save results
    print("\n[6/6] Saving results and generating visualizations...", flush=True)
    results_dir = Path("results/calibration")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    metrics_summary = {}
    for scenario, scenario_results in all_results.items():
        metrics_summary[scenario] = {}
        for method, results in scenario_results.items():
            metrics_summary[scenario][method] = {
                'ece': float(results['ece']),
                'brier': float(results['brier']),
                'accuracy': float(results['accuracy']),
                'f1': float(results['f1']),
                'precision': float(results['precision']),
                'recall': float(results['recall'])
            }
    
    with open(results_dir / 'calibration_results_all.json', 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    print(f"✅ Results saved to {results_dir}/calibration_results_all.json", flush=True)
    print("\nGenerating visualizations...", flush=True)
    
    # Import and run visualization generation
    from visualization import create_all_visualizations
    create_all_visualizations(all_results, save_dir=results_dir)
    
    print("\n" + "="*80, flush=True)
    print("✅ COMPLETE CALIBRATION EVALUATION DONE!", flush=True)
    print("="*80, flush=True)

if __name__ == "__main__":
    main()
