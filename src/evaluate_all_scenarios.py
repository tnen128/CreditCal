#!/usr/bin/env python3
"""
Comprehensive Evaluation - Lee et al. (2023) Replication
Implements all 5 evaluation scenarios from Section 4
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

from models_replication import CreditLSTMPaper

# ============================================================================
# CONFIGURATION (MINIMAL TEST VERSION)
# ============================================================================
DATA_FILE = 'data/test_2006q1.csv'
GLOBAL_ROUNDS = 1  # MINIMAL TEST (Paper: 100)
LOCAL_EPOCHS = 1   # MINIMAL TEST (Paper: 10)
BATCH_SIZE = 128
LR = 0.01
MOMENTUM = 0.9
DECAY = 0.0001
MAX_SEQ_LEN = 60

DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}", flush=True)

# ============================================================================
# DATASET CLASS
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
# TRAINING & EVALUATION FUNCTIONS
# ============================================================================

def train_model(model, dataloader, epochs, device, verbose=True):
    """Train a single model"""
    model.train()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=DECAY)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        for sequences, labels, lengths in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        
        if verbose and n_batches > 0:
            print(f"      Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batches:.4f}")
    
    return model

def evaluate_model(model, dataloader, device):
    """Evaluate model and return all 4 metrics"""
    model.eval()
    model.to(device)
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for sequences, labels, lengths in dataloader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.numpy().flatten())
    
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return {'accuracy': accuracy, 'recall': recall, 'precision': precision, 'f1': f1}

def federated_averaging(global_model, client_models):
    """FedAvg aggregation"""
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack([c.state_dict()[key].float() for c in client_models], 0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model

# ============================================================================
# EVALUATION SCENARIOS
# ============================================================================

def scenario_1_local_models(clients_train, clients_test, feature_cols, scaler, input_dim):
    """
    Scenario 1: Local Model
    Each FI trains independently on their own data
    """
    print("\n" + "="*80)
    print("SCENARIO 1: LOCAL MODELS (Independent Training)")
    print("="*80)
    
    results = {}
    
    for client_id, (client_name, train_data) in enumerate(clients_train.items()):
        print(f"\nTraining Local Model for {client_name}...")
        
        # Create datasets
        train_dataset = FixedEncodingDataset(train_data, feature_cols, scaler)
        test_dataset = FixedEncodingDataset(clients_test[client_name], feature_cols, scaler)
        
        if len(train_dataset) == 0:
            print(f"  Skipped (no training data)")
            continue
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        
        # Train local model
        model = CreditLSTMPaper(input_dim=input_dim)
        model = train_model(model, train_loader, LOCAL_EPOCHS, DEVICE, verbose=False)
        
        # Evaluate
        metrics = evaluate_model(model, test_loader, DEVICE)
        results[client_name] = metrics
        print(f"  Acc: {metrics['accuracy']:.4f}, Rec: {metrics['recall']:.4f}, Prec: {metrics['precision']:.4f}, F1: {metrics['f1']:.4f}")
    
    # Compute average
    avg_metrics = {
        'accuracy': np.mean([m['accuracy'] for m in results.values()]),
        'recall': np.mean([m['recall'] for m in results.values()]),
        'precision': np.mean([m['precision'] for m in results.values()]),
        'f1': np.mean([m['f1'] for m in results.values()])
    }
    
    print(f"\n  AVERAGE: Acc: {avg_metrics['accuracy']:.4f}, Rec: {avg_metrics['recall']:.4f}, Prec: {avg_metrics['precision']:.4f}, F1: {avg_metrics['f1']:.4f}")
    
    return results, avg_metrics

def scenario_2_central_model(train_df, test_df, feature_cols, scaler, input_dim):
    """
    Scenario 2: Central Model
    Single model trained on all data combined
    """
    print("\n" + "="*80)
    print("SCENARIO 2: CENTRAL MODEL (All Data Combined)")
    print("="*80)
    
    train_dataset = FixedEncodingDataset(train_df, feature_cols, scaler)
    test_dataset = FixedEncodingDataset(test_df, feature_cols, scaler)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Train central model
    model = CreditLSTMPaper(input_dim=input_dim)
    model = train_model(model, train_loader, LOCAL_EPOCHS, DEVICE, verbose=True)
    
    # Evaluate
    metrics = evaluate_model(model, test_loader, DEVICE)
    print(f"\n  CENTRAL: Acc: {metrics['accuracy']:.4f}, Rec: {metrics['recall']:.4f}, Prec: {metrics['precision']:.4f}, F1: {metrics['f1']:.4f}")
    
    return metrics

def scenario_3_federated_learning(clients_train, clients_test, feature_cols, scaler, input_dim, exclude_top_n=0):
    """
    Scenario 3: Federated Learning
    a) All clients (n)
    b) Exclude top 1 client (n-1)
    c) Exclude top 2 clients (n-2)
    """
    scenario_name = f"FL (n-{exclude_top_n})" if exclude_top_n > 0 else "FL (n)"
    
    print("\n" + "="*80)
    print(f"SCENARIO 3{chr(97+exclude_top_n)}: {scenario_name}")
    print("="*80)
    
    # Sort clients by size and exclude if needed
    client_sizes = sorted([(name, len(data)) for name, data in clients_train.items()], key=lambda x: x[1], reverse=True)
    active_clients = [name for name, _ in client_sizes[exclude_top_n:]]
    
    print(f"Active clients: {len(active_clients)}")
    if exclude_top_n > 0:
        print(f"Excluded: {[name for name, _ in client_sizes[:exclude_top_n]]}")
    
    # Initialize global model
    global_model = CreditLSTMPaper(input_dim=input_dim)
    
    # Federated training
    for round_num in range(GLOBAL_ROUNDS):
        print(f"\n  Round {round_num+1}/{GLOBAL_ROUNDS}")
        client_models = []
        
        for client_name in active_clients:
            train_data = clients_train[client_name]
            train_dataset = FixedEncodingDataset(train_data, feature_cols, scaler)
            
            if len(train_dataset) == 0:
                continue
            
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)
            
            # Train client
            client_model = copy.deepcopy(global_model)
            client_model = train_model(client_model, train_loader, LOCAL_EPOCHS, DEVICE, verbose=False)
            client_models.append(client_model)
        
        # Aggregate
        if len(client_models) > 0:
            global_model = federated_averaging(global_model, client_models)
    
    # Evaluate on each client's test set
    results = {}
    for client_name in active_clients:
        test_dataset = FixedEncodingDataset(clients_test[client_name], feature_cols, scaler)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        metrics = evaluate_model(global_model, test_loader, DEVICE)
        results[client_name] = metrics
    
    # Average metrics
    avg_metrics = {
        'accuracy': np.mean([m['accuracy'] for m in results.values()]),
        'recall': np.mean([m['recall'] for m in results.values()]),
        'precision': np.mean([m['precision'] for m in results.values()]),
        'f1': np.mean([m['f1'] for m in results.values()])
    }
    
    print(f"\n  AVERAGE: Acc: {avg_metrics['accuracy']:.4f}, Rec: {avg_metrics['recall']:.4f}, Prec: {avg_metrics['precision']:.4f}, F1: {avg_metrics['f1']:.4f}")
    
    return results, avg_metrics

# ============================================================================
# MAIN EVALUATION
# ============================================================================

def main():
    print("="*80)
    print("COMPREHENSIVE EVALUATION - Lee et al. (2023)")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    print("\n[1/4] Loading dataset...")
    df = pd.read_csv(DATA_FILE)
    df['year'] = df['MONTHLY_REPORTING_PERIOD'].astype(str).str[:4].astype(int)
    
    # Temporal split (2006-2008 train, 2009 test)
    train_df = df[df['year'].isin([2006, 2007, 2008])].copy()
    test_df = df[df['year'] == 2009].copy()
    
    print(f"  Train (2006-2008): {len(train_df):,} rows")
    print(f"  Test (2009): {len(test_df):,} rows")
    
    # Prepare features
    print("\n[2/4] Preparing features...")
    exclude = ['LOAN_SEQUENCE_NUMBER', 'MONTHLY_REPORTING_PERIOD', 'DEFAULT_LABEL', 
               'ZERO_BALANCE_CODE', 'year', 'month', 'SELLER_NAME']
    
    # Force numeric
    numeric_cols = ['CREDIT_SCORE', 'ORIG_INTEREST_RATE', 'ORIG_UPB', 'ORIG_LOAN_TERM', 
                   'LTV', 'MI_PERCENT', 'CURRENT_ACTUAL_UPB', 'LOAN_AGE', 
                   'REMAINING_MONTHS_TO_LEGAL_MATURITY', 'HPI', 'UNEMPLOYMENT_RATE', 
                   'INTEREST_RATE_30YR', 'DELINQUENCY_RATE_FRED', 'ELTV', 'NUM_UNITS', 
                   'NUM_BORROWERS', 'CURRENT_LOAN_DELINQUENCY_STATUS', 'DTI', 
                   'UNEMPLOYMENT_AT_ORIGINATION', 'CHARGEOFF_RATE']
    
    for col in numeric_cols:
        if col in train_df.columns:
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0)
    
    # One-hot encode
    potential_features = [c for c in train_df.columns if c not in exclude]
    cat_cols = train_df[potential_features].select_dtypes(include=['object']).columns.tolist()
    
    train_encoded = pd.get_dummies(train_df, columns=cat_cols, dummy_na=True)
    test_encoded = pd.get_dummies(test_df, columns=cat_cols, dummy_na=True)
    
    feature_cols = [c for c in train_encoded.columns if c not in exclude]
    print(f"  Features: {len(feature_cols)}")
    
    # Fit scaler
    scaler = StandardScaler()
    scaler.fit(train_encoded[feature_cols].fillna(0))
    
    # Partition by SELLER_NAME
    print("\n[3/4] Partitioning clients...")
    clients_train = {name: group for name, group in train_encoded.groupby('SELLER_NAME')}
    clients_test = {name: group for name, group in test_encoded.groupby('SELLER_NAME')}
    
    print(f"  Total FIs: {len(clients_train)}")
    
    # Run evaluations
    print("\n[4/4] Running evaluation scenarios...")
    
    all_results = {}
    
    # Scenario 1: Local Models
    local_results, local_avg = scenario_1_local_models(clients_train, clients_test, feature_cols, scaler, len(feature_cols))
    all_results['Local'] = local_avg
    
    # Scenario 2: Central Model
    central_results = scenario_2_central_model(train_encoded, test_encoded, feature_cols, scaler, len(feature_cols))
    all_results['Central'] = central_results
    
    # Scenario 3a: FL (n)
    fl_n_results, fl_n_avg = scenario_3_federated_learning(clients_train, clients_test, feature_cols, scaler, len(feature_cols), exclude_top_n=0)
    all_results['FL (n)'] = fl_n_avg
    
    # Scenario 3b: FL (n-1)
    fl_n1_results, fl_n1_avg = scenario_3_federated_learning(clients_train, clients_test, feature_cols, scaler, len(feature_cols), exclude_top_n=1)
    all_results['FL (n-1)'] = fl_n1_avg
    
    # Scenario 3c: FL (n-2)
    fl_n2_results, fl_n2_avg = scenario_3_federated_learning(clients_train, clients_test, feature_cols, scaler, len(feature_cols), exclude_top_n=2)
    all_results['FL (n-2)'] = fl_n2_avg
    
    # Final results table
    print("\n" + "="*80)
    print("FINAL RESULTS TABLE")
    print("="*80)
    print(f"{'Scenario':<15} {'Accuracy':<10} {'Recall':<10} {'Precision':<10} {'F1':<10}")
    print("-"*80)
    for scenario, metrics in all_results.items():
        print(f"{scenario:<15} {metrics['accuracy']:.4f}     {metrics['recall']:.4f}     {metrics['precision']:.4f}     {metrics['f1']:.4f}")
    
    # Save results
    with open('evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to evaluation_results.json")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
