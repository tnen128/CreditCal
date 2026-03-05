#!/usr/bin/env python3
"""
COMPLETE FL EVALUATION - All Scenarios
=======================================
Implements all FL scenarios from the paper:

1. Local (L): Institution-specific models
2. Central (C): Single model on pooled data  
3. FL(n): Federated Learning with all n institutions
4. FL(n-1): Exclude LARGEST institution (#1) from training, test on it
5. FL(n-2): Exclude 2 LARGEST institutions (#1 & #2) from training, test on them

All scenarios use institution-aware calibration for fair comparison.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import json
import joblib
from pathlib import Path
from scipy.special import expit as sigmoid
import copy
import time
from itertools import combinations
import warnings

# Suppress sklearn feature name warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='X does not have valid feature names')

import sys
sys.path.append('src')

from models import CreditLSTMPaper
from calibration import get_calibrator, compute_ece, compute_brier_score

# ============================================================================
# CONFIGURATION
# ============================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# PRODUCTION PARAMETERS
EPOCHS = 1  # Production: 10 epochs
FL_ROUNDS = 1  # Production: 30 rounds
MAX_LOANS_PER_INSTITUTION = 50
N_INSTITUTIONS_TRIAL = 14  # All 14 institutions!
BATCH_SIZE = 256
LR = 0.01
MOMENTUM = 0.9
DECAY = 0.0001
MAX_SEQ_LEN = 60

DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = Path("results/experiments")
CALIBRATION_METHODS = ['temperature', 'platt', 'isotonic', 'beta']

print("="*80)
print("COMPLETE FL EVALUATION: L + C + FL(n) + FL(n-1) + FL(n-2)")
print("="*80)
print(f"Device: {DEVICE}")
print(f"Institutions: {N_INSTITUTIONS_TRIAL}")
print(f"Epochs: {EPOCHS}, FL Rounds: {FL_ROUNDS}")
print(f"Output: {OUTPUT_DIR}")
print("="*80)

# ============================================================================
# DATASET
# ============================================================================
class TrialDataset(Dataset):
    def __init__(self, df, feature_cols, scaler, max_loans=None):
        if max_loans:
            unique_loans = df['LOAN_SEQUENCE_NUMBER'].unique()[:max_loans]
            df = df[df['LOAN_SEQUENCE_NUMBER'].isin(unique_loans)]
        
        self.df = df
        self.feature_cols = feature_cols
        self.scaler = scaler
        self.loan_ids = df['LOAN_SEQUENCE_NUMBER'].unique()
        
    def __len__(self):
        return len(self.loan_ids)
    
    def __getitem__(self, idx):
        loan_id = self.loan_ids[idx]
        loan_data = self.df[self.df['LOAN_SEQUENCE_NUMBER'] == loan_id].sort_values('MONTHLY_REPORTING_PERIOD')
        
        features = loan_data[self.feature_cols].values
        features_scaled = self.scaler.transform(features)
        label = loan_data['DEFAULT_LABEL'].iloc[0]
        seq_len = len(loan_data)
        
        return torch.tensor(features_scaled, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), seq_len

def collate_fn(batch):
    sequences, labels, lengths = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)
    labels = torch.stack(labels)
    
    max_len = min(max(lengths), MAX_SEQ_LEN)
    padded_sequences = []
    
    for seq, length in zip(sequences, lengths):
        if length > max_len:
            seq = seq[:max_len]
        else:
            padding = torch.zeros(max_len - length, seq.shape[1])
            seq = torch.cat([seq, padding], dim=0)
        padded_sequences.append(seq)
    
    return torch.stack(padded_sequences), labels, torch.clamp(lengths, max=max_len)

# ============================================================================
# TRAINING
# ============================================================================
def train_simple(model, train_loader, device, epochs=1):
    model.train()
    model.to(device)
    
    total_pos = sum(labels.sum().item() for _, labels, _ in train_loader)
    total_neg = sum((1 - labels).sum().item() for _, labels, _ in train_loader)
    pos_weight = torch.tensor(total_neg / max(total_pos, 1)).to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=DECAY)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    for epoch in range(epochs):
        for sequences, labels, lengths in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(sequences, lengths).flatten()
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    
    return model

def get_predictions(model, dataloader, device):
    model.eval()
    model.to(device)  # ← FIX: Ensure model is on correct device
    all_logits, all_labels = [], []
    
    with torch.no_grad():
        for sequences, labels, lengths in dataloader:
            sequences = sequences.to(device)
            logits = model(sequences, lengths).flatten()
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy().flatten())
    
    return np.concatenate(all_logits), np.concatenate(all_labels)

def compute_metrics(probs, labels):
    preds = (probs >= 0.5).astype(int)
    return {
        'ece': float(compute_ece(probs, labels)),
        'brier': float(compute_brier_score(probs, labels)),
        'accuracy': float(accuracy_score(labels, preds)),
        'f1': float(f1_score(labels, preds, zero_division=0))
    }

def evaluate_institution_aware(model, val_loaders_dict, test_loaders_dict, device, save_dir):
    """
    CRITICAL: Institution-aware calibration for fair comparison.
    
    For each institution:
    - Fit calibrator on that institution's val set
    - Apply to that institution's test set
    - Save probabilities to CSV files
    """
    import pandas as pd
    save_dir = Path(save_dir)
    results = {}
    
    for institution in val_loaders_dict.keys():
        inst_dir = save_dir / institution.replace("/", "_").replace(" ", "_")
        inst_dir.mkdir(parents=True, exist_ok=True)
        (inst_dir / "calibrators").mkdir(exist_ok=True)
        
        # Get predictions for THIS institution only
        val_logits, val_labels = get_predictions(model, val_loaders_dict[institution], device)
        test_logits, test_labels = get_predictions(model, test_loaders_dict[institution], device)
        
        # Prepare CSV dataframe
        probs_df = pd.DataFrame({
            'true_label': test_labels,
            'logits': test_logits
        })
        
        # Uncalibrated
        test_probs = sigmoid(test_logits)
        probs_df['prob_uncalibrated'] = test_probs
        inst_results = {'uncalibrated': compute_metrics(test_probs, test_labels)}
        
        # Calibrate using THIS institution's validation set
        for method in CALIBRATION_METHODS:
            try:
                calibrator = get_calibrator(method)
                calibrator.fit(val_logits, val_labels)
                joblib.dump(calibrator, inst_dir / "calibrators" / f"{method}.pkl")
                test_probs_cal = calibrator.transform(test_logits)
            except Exception as e:
                # Falls back to uncalibrated if val set has only one class (small datasets)
                print(f"    ⚠️  {method} calibration skipped for {institution}: {e}")
                test_probs_cal = sigmoid(test_logits)
            probs_df[f'prob_{method}'] = test_probs_cal
            inst_results[method] = compute_metrics(test_probs_cal, test_labels)

        # Save probabilities to CSV
        probs_df.to_csv(inst_dir / "predictions.csv", index=False)
        
        results[institution] = inst_results
    
    return results

# ============================================================================
# INCREMENTAL RESULTS SAVING
# ============================================================================
def save_results_incremental(results, output_dir):
    """Save results to JSON after each scenario completes."""
    with open(output_dir / "complete_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  💾 Results saved to {output_dir / 'complete_results.json'}")

# ============================================================================
# FEDAVG
# ============================================================================
def fedavg(global_model, local_models, client_weights):
    global_dict = global_model.state_dict()
    
    for key in global_dict.keys():
        global_dict[key] = sum(
            client_weights[i] * local_models[i].state_dict()[key]
            for i in range(len(local_models))
        )
    
    global_model.load_state_dict(global_dict)
    return global_model

# ============================================================================
# MAIN
# ============================================================================
def main():
    start_time = time.time()
    
    # Load data
    print("\n[1/6] Loading data...")
    with open("data/preprocessed_strict/metadata.json") as f:
        metadata = json.load(f)
    
    feature_cols = metadata['feature_cols']
    input_dim = len(feature_cols)
    all_institutions = metadata['top_14_institutions'][:N_INSTITUTIONS_TRIAL]
    
    train_df = pd.read_csv("data/preprocessed_strict/train.csv")
    val_df = pd.read_csv("data/preprocessed_strict/val.csv")
    test_df = pd.read_csv("data/preprocessed_strict/test.csv")
    scaler = joblib.load("data/preprocessed_strict/scaler.pkl")
    
    # Filter
    train_df = train_df[train_df['SELLER_NAME'].isin(all_institutions)]
    val_df = val_df[val_df['SELLER_NAME'].isin(all_institutions)]
    test_df = test_df[test_df['SELLER_NAME'].isin(all_institutions)]
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    
    # Create per-institution datasets
    train_datasets = {
        inst: TrialDataset(train_df[train_df['SELLER_NAME'] == inst], feature_cols, scaler, MAX_LOANS_PER_INSTITUTION)
        for inst in all_institutions
    }
    val_datasets = {
        inst: TrialDataset(val_df[val_df['SELLER_NAME'] == inst], feature_cols, scaler, MAX_LOANS_PER_INSTITUTION // 2)
        for inst in all_institutions
    }
    test_datasets = {
        inst: TrialDataset(test_df[test_df['SELLER_NAME'] == inst], feature_cols, scaler, MAX_LOANS_PER_INSTITUTION // 2)
        for inst in all_institutions
    }
    
    # ========================================================================
    # SCENARIO 1: LOCAL MODELS
    # ========================================================================
    print(f"\n[2/6] LOCAL MODELS")
    print("-" * 80)
    
    results['Local'] = {}
    
    for inst in all_institutions:
        print(f"  Training {inst}...")
        
        inst_dir = OUTPUT_DIR / "Local" / inst.replace("/", "_").replace(" ", "_")
        inst_dir.mkdir(parents=True, exist_ok=True)
        
        train_loader = DataLoader(train_datasets[inst], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_datasets[inst], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_datasets[inst], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        
        model = CreditLSTMPaper(input_dim=input_dim)
        model = train_simple(model, train_loader, DEVICE, EPOCHS)
        torch.save(model.state_dict(), inst_dir / "model.pth")
        
        # Institution-aware calibration (trivial for Local - only one institution)
        inst_results = evaluate_institution_aware(
            model,
            {inst: val_loader},
            {inst: test_loader},
            DEVICE,
            inst_dir
        )
        results['Local'][inst] = inst_results[inst]
    
    # Save results after Local models complete
    save_results_incremental(results, OUTPUT_DIR)
    
    # ========================================================================
    # SCENARIO 2: CENTRAL MODEL
    # ========================================================================
    print(f"\n[3/6] CENTRAL MODEL")
    print("-" * 80)
    
    central_dir = OUTPUT_DIR / "Central"
    central_dir.mkdir(parents=True, exist_ok=True)
    
    train_pooled = ConcatDataset(list(train_datasets.values()))
    train_loader = DataLoader(train_pooled, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
   # Create val/test loaders per institution for institution-aware calibration
    val_loaders = {
        inst: DataLoader(val_datasets[inst], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        for inst in all_institutions
    }
    test_loaders = {
        inst: DataLoader(test_datasets[inst], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        for inst in all_institutions
    }
    
    print(f"  Training central model...")
    central_model = CreditLSTMPaper(input_dim=input_dim)
    central_model = train_simple(central_model, train_loader, DEVICE, EPOCHS)
    torch.save(central_model.state_dict(), central_dir / "model.pth")
    
    # Institution-aware calibration
    results['Central'] = evaluate_institution_aware(central_model, val_loaders, test_loaders, DEVICE, central_dir)
    
    # Save results after Central model completes
    save_results_incremental(results, OUTPUT_DIR)
    
    # ========================================================================
    # SCENARIO 3: FL(n) - All institutions
    # ========================================================================
    print(f"\n[4/6] FL(n) - All {len(all_institutions)} institutions")
    print("-" * 80)
    
    fl_n_dir = OUTPUT_DIR / "FL_n"
    fl_n_dir.mkdir(parents=True, exist_ok=True)
    
    global_model = CreditLSTMPaper(input_dim=input_dim)
    
    client_sizes = [len(train_datasets[inst]) for inst in all_institutions]
    total_size = sum(client_sizes)
    client_weights = [size / total_size for size in client_sizes]
    
    for round_idx in range(FL_ROUNDS):
        print(f"  Round {round_idx + 1}/{FL_ROUNDS}")
        
        local_models = []
        for inst in all_institutions:
            local_model = copy.deepcopy(global_model)
            train_loader = DataLoader(train_datasets[inst], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
            local_model = train_simple(local_model, train_loader, DEVICE, EPOCHS)
            local_models.append(local_model)
        
        global_model = fedavg(global_model, local_models, client_weights)
        torch.save(global_model.state_dict(), fl_n_dir / f"round_{round_idx}_model.pth")
    
    torch.save(global_model.state_dict(), fl_n_dir / "global_model.pth")
    
    # Institution-aware calibration for FL(n)
    results['FL_n'] = evaluate_institution_aware(global_model, val_loaders, test_loaders, DEVICE, fl_n_dir)
    
    # Save results after FL(n) completes
    save_results_incremental(results, OUTPUT_DIR)
    
    # ========================================================================
    # SCENARIO 4: FL(n-1) - Exclude LARGEST institution only
    # ========================================================================
    print(f"\n[5/6] FL(n-1) - Exclude largest institution")
    print("-" * 80)
    
    results['FL_n_minus_1'] = {}
    
    # Exclude ONLY the largest institution (#1 = Wells Fargo)
    excluded_inst = all_institutions[0]  # First institution is largest
    print(f"\n  Excluding largest institution: {excluded_inst}")
    
    fl_n1_dir = OUTPUT_DIR / "FL_n_minus_1"
    fl_n1_dir.mkdir(parents=True, exist_ok=True)
    
    # Train on all EXCEPT largest institution
    training_institutions = all_institutions[1:]  # All except first
    
    global_model = CreditLSTMPaper(input_dim=input_dim)
    
    client_sizes = [len(train_datasets[inst]) for inst in training_institutions]
    total_size = sum(client_sizes)
    client_weights = [size / total_size for size in client_sizes]
    
    for round_idx in range(FL_ROUNDS):
        if round_idx % 5 == 0:  # Print every 5 rounds
            print(f"  Round {round_idx + 1}/{FL_ROUNDS}")
        
        local_models = []
        for inst in training_institutions:
            local_model = copy.deepcopy(global_model)
            train_loader = DataLoader(train_datasets[inst], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
            local_model = train_simple(local_model, train_loader, DEVICE, EPOCHS)
            local_models.append(local_model)
        
        global_model = fedavg(global_model, local_models, client_weights)
    
    torch.save(global_model.state_dict(), fl_n1_dir / "global_model.pth")
    
    # Evaluate ONLY on excluded (largest) institution
    excluded_results = evaluate_institution_aware(
        global_model,
        {excluded_inst: val_loaders[excluded_inst]},
        {excluded_inst: test_loaders[excluded_inst]},
        DEVICE,
        fl_n1_dir
    )
    results['FL_n_minus_1'] = excluded_results[excluded_inst]
    
    # Save results after FL(n-1) completes
    save_results_incremental(results, OUTPUT_DIR)
    
    # ========================================================================
    # SCENARIO 5: FL(n-2) - Exclude 2 LARGEST institutions
    # ========================================================================
    print(f"\n[6/6] FL(n-2) - Exclude 2 largest institutions")
    print("-" * 80)
    
    results['FL_n_minus_2'] = {}
    
    # Exclude ONLY the 2 largest institutions (#1 & #2)
    excluded_pair = all_institutions[:2]  # First 2 institutions are largest
    print(f"\n  Excluding 2 largest: {excluded_pair[0]} + {excluded_pair[1]}")
    
    fl_n2_dir = OUTPUT_DIR / "FL_n_minus_2"
    fl_n2_dir.mkdir(parents=True, exist_ok=True)
    
    # Train on all EXCEPT 2 largest
    training_institutions = all_institutions[2:]  # All except first 2
    
    if len(training_institutions) < 2:
        print("  Skipping (need at least 2 training institutions)")
    else:
        global_model = CreditLSTMPaper(input_dim=input_dim)
        
        client_sizes = [len(train_datasets[inst]) for inst in training_institutions]
        total_size = sum(client_sizes)
        client_weights = [size / total_size for size in client_sizes]
        
        for round_idx in range(FL_ROUNDS):
            if round_idx % 5 == 0:  # Print every 5 rounds
                print(f"  Round {round_idx + 1}/{FL_ROUNDS}")
            
            local_models = []
            for inst in training_institutions:
                local_model = copy.deepcopy(global_model)
                train_loader = DataLoader(train_datasets[inst], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
                local_model = train_simple(local_model, train_loader, DEVICE, EPOCHS)
                local_models.append(local_model)
            
            global_model = fedavg(global_model, local_models, client_weights)
        
        torch.save(global_model.state_dict(), fl_n2_dir / "global_model.pth")
        
        # Evaluate ONLY on 2 excluded (largest) institutions
        excluded_val = {inst: val_loaders[inst] for inst in excluded_pair}
        excluded_test = {inst: test_loaders[inst] for inst in excluded_pair}
        
        excluded_results = evaluate_institution_aware(global_model, excluded_val, excluded_test, DEVICE, fl_n2_dir)
        results['FL_n_minus_2'] = excluded_results
    
    # Save results after FL(n-2) completes
    save_results_incremental(results, OUTPUT_DIR)
    
    # ========================================================================
    # FINAL SAVE & SUMMARY
    # ========================================================================
    with open(OUTPUT_DIR / "complete_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*80)
    print("✅ COMPLETE FL EVALUATION FINISHED!")
    print("="*80)
    print(f"Time: {elapsed/60:.1f} minutes")
    

if __name__ == "__main__":
    main()
