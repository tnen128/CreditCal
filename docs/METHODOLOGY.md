# Detailed Methodology

## How to Run Experiments

### Quick Start (5 Minutes)

**For testing the pipeline with 2006Q1 data:**

```bash
# 1. Ensure you're in the repository root
cd fl-credit-risk-assessment

# 2. Make sure data exists
ls data/test_2006q1.csv  # Should exist

# 3. Run calibration evaluation (uses config/config.yaml)
python src/evaluate_calibration.py
```

**Expected Output:**
- Console shows progress through 5 scenarios
- Results saved to `results/calibration/calibration_results_all.json`
- 6 visualizations generated in `results/calibration/*.png`
- Runtime: ~2 minutes

### Full Experiment Setup

**Step 1: Download Freddie Mac Data**

1. Visit: https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset
2. Create a free account
3. Download Single-Family Loan-Level Dataset for 2006-2009:
   - Origination files: `sample_orig_YYYY.txt`
   - Performance files: `sample_svcg_YYYY.txt`
   - Quarters: Q1-Q4 for years 2006, 2007, 2008, 2009

**Step 2: Organize Data**

Place downloaded files in `data/raw/`:
```
data/raw/
├── sample_orig_2006.txt
├── sample_svcg_2006.txt
├── sample_orig_2007.txt
├── sample_svcg_2007.txt
├── sample_orig_2008.txt
├── sample_svcg_2008.txt
├── sample_orig_2009.txt
└── sample_svcg_2009.txt
```

**Step 3: Configure Experiment**

Edit `config/config.yaml`:

```yaml
# For full experiment (10-24 hours)
training:
  global_rounds: 100  # Change from 1
  local_epochs: 10    # Change from 1

data:
  quick_test: false   # Change from true
```

**Step 4: Run Preprocessing**

```bash
# Preprocess all data (2006-2009)
python src/preprocess.py

# This will:
# - Load all quarters
# - Merge macroeconomic variables
# - Engineer features
# - Create replication_dataset_strict.csv
# - Runtime: 3-6 hours
```

**Step 5: Run Evaluation**

```bash
# Option A: All scenarios + calibration
python src/evaluate_calibration.py

# Option B: Just scenario comparison (no calibration)
python src/evaluate_all_scenarios.py
```

### Configuration Guide

All parameters are in `config/config.yaml`. Here are the key settings:

#### Training Parameters

```yaml
training:
  global_rounds: 100      # Number of FL rounds (1 for testing, 100 for final)
  local_epochs: 10        # Epochs per client per round (1 for testing, 10 for final)
  batch_size: 128         # Batch size (reduce if out of memory)
  learning_rate: 0.01     # SGD learning rate
  momentum: 0.9           # SGD momentum
  weight_decay: 0.0001    # L2 regularization
```

#### Data Parameters

```yaml
data:
  quick_test: true        # If true, uses only 2006Q1 data
  test_quarters: ["2006Q1"]  # Quarters for quick test
  train_years: [2006, 2007]  # Training period
  val_year: 2008             # Validation (for calibration)
  test_year: 2009            # Testing period
```

#### Scenario Selection

```yaml
federated_learning:
  scenarios:
    local: true       # Scenario 1: Independent local models
    central: true     # Scenario 2: Centralized model
    fl_n: true        # Scenario 3a: FL with all clients
    fl_n_minus_1: true  # Scenario 3b: FL without biggest client
    fl_n_minus_2: true  # Scenario 3c: FL without top 2 clients
```

#### Calibration Methods

```yaml
calibration:
  methods:
    platt: true        # Platt Scaling
    isotonic: true     # Isotonic Regression
    temperature: true  # Temperature Scaling (recommended)
    beta: true         # Beta Calibration
```

### Common Configurations

#### 1. Quick Testing (2 minutes)
```yaml
training:
  global_rounds: 1
  local_epochs: 1
data:
  quick_test: true
```

#### 2. Medium Run (1-2 hours)
```yaml
training:
  global_rounds: 10
  local_epochs: 5
data:
  quick_test: false
```

#### 3. Full Replication (10-24 hours)
```yaml
training:
  global_rounds: 100
  local_epochs: 10
data:
  quick_test: false
```

### Monitoring Progress

**During training, you'll see:**
```
[3/6] Scenario 1: Local Models...
    Uncalibrated - ECE: 0.3220, F1: 0.9723
    Platt - ECE: 0.0102 (Δ -0.3118)
    Isotonic - ECE: 0.0219 (Δ -0.3001)
    Temperature - ECE: 0.0102 (Δ -0.3118)
    Beta - ECE: 0.0102 (Δ -0.3118)
```

**Progress indicators:**
- `[X/6]`: Current scenario being evaluated
- `ECE`: Expected Calibration Error (lower is better)
- `Δ`: Change from uncalibrated (negative = improvement)

### Understanding Results

**Results Location:**
```
results/
├── calibration/
│   ├── 1_heatmaps_comparison.png      # 3-panel heatmap
│   ├── 2_reliability_diagrams_all.png # Calibration curves
│   ├── 3_overall_performance.png      # Bar chart
│   ├── 4_improvement_matrix.png       # ECE improvements
│   ├── 5_method_ranking.png           # Best methods
│   ├── 6_brier_comparison.png         # Probability accuracy
│   └── calibration_results_all.json   # All metrics
└── evaluation/
    └── evaluation_results.json        # Scenario comparison
```

**Key Metrics:**

1. **ECE (Expected Calibration Error)**: 0.0-1.0 (lower is better)
   - Excellent: < 0.01
   - Good: < 0.05
   - Poor: > 0.10

2. **Brier Score**: 0.0-1.0 (lower is better)
   - Excellent: < 0.05
   - Good: < 0.10
   - Poor: > 0.20

3. **F1 Score**: 0.0-1.0 (higher is better)
   - Excellent: > 0.95
   - Good: > 0.90
   - Fair: > 0.80

**Expected Results (2006Q1 Quick Test):**
| Metric | Before Calib | After Calib | Improvement |
|--------|--------------|-------------|-------------|
| ECE    | ~0.30        | ~0.01       | 96-98% ⬇️   |
| Brier  | ~0.16        | ~0.07       | 50-60% ⬇️   |
| F1     | ~0.96        | ~0.96       | Unchanged   |

### Troubleshooting

**Issue: Out of memory**
```yaml
# In config.yaml, reduce batch size:
training:
  batch_size: 64  # or 32
```

**Issue: Too slow**
```yaml
# Reduce rounds/epochs:
training:
  global_rounds: 10  # instead of 100
  local_epochs: 5    # instead of 10
```

**Issue: Data not found**
- Check `data/test_2006q1.csv` exists for quick test
- Check `data/raw/*.txt` exists for full run
- Ensure file paths in config match your setup

**Issue: Import errors**
```bash
# Reinstall dependencies:
pip install -r requirements.txt
```

### Performance Optimization

**GPU Acceleration:**
- CUDA: Automatically detected if available
- MPS (Apple Silicon): Automatically detected
- CPU: Fallback option

**Speed Tips:**
1. Use GPU if available (10-100x faster)
2. Increase batch size (if memory allows)
3. Use quick_test mode for development
4. Disable unnecessary scenarios in config

### Experiment Checklist

Before running full experiment:

- [ ] Downloaded all Freddie Mac data (2006-2009)
- [ ] Placed data in `data/raw/` directory
- [ ] Updated `config/config.yaml` with desired parameters
- [ ] Tested with quick_test mode first
- [ ] Have 10-24 hours available for full run
- [ ] Have sufficient disk space (~10GB for data + results)
- [ ] Installed all requirements (`pip install -r requirements.txt`)

---

## Overview

This document provides a comprehensive description of the federated learning methodology implemented in this project, strictly following Lee et al. (2023).

## Data Preprocessing

### 1. Data Source
- **Dataset**: Freddie Mac Single-Family Loan-Level Dataset
- **Time Period**: 2006-2009 (origination and performance data)
- **Download**: https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset

### 2. Data Schema

#### Origination File (27 variables)
```python
ORIG_COLS = [
    'CREDIT_SCORE', 'FIRST_PAYMENT_DATE', 'FIRST_TIME_HOMEBUYER',
    'MATURITY_DATE', 'MSA', 'MI_PERCENT', 'NUM_UNITS', 'OCCUPANCY_STATUS',
    'ORIG_CLTV', 'DTI', 'ORIG_UPB', 'LTV', 'ORIG_INTEREST_RATE',
    'CHANNEL', 'PPM_FLAG', 'AMORTIZATION_TYPE', 'PROPERTY_STATE',
    'PROPERTY_TYPE', 'POSTAL_CODE', 'LOAN_SEQUENCE_NUMBER', 'LOAN_PURPOSE',
    'ORIG_LOAN_TERM', 'NUM_BORROWERS', 'SELLER_NAME', 'SERVICER_NAME',
    'SUPER_CONFORMING_FLAG', 'PRE_HARP_LOAN_SEQUENCE_NUMBER'
]
```

#### Performance File (28 variables)
```python
PERF_COLS = [
    'LOAN_SEQUENCE_NUMBER', 'MONTHLY_REPORTING_PERIOD', 'CURRENT_ACTUAL_UPB',
    'CURRENT_LOAN_DELINQUENCY_STATUS', 'LOAN_AGE', 
    'REMAINING_MONTHS_TO_LEGAL_MATURITY', 'DEFECT_SETTLEMENT_DATE',
    'MODIFICATION_FLAG', 'ZERO_BALANCE_CODE', 'ZERO_BALANCE_EFFECTIVE_DATE',
    'CURRENT_INTEREST_RATE', 'CURRENT_DEFERRED_UPB', 'DDLPI', 'MI_RECOVERIES',
    'NET_SALE_PROCEEDS', 'NON_MI_RECOVERIES', 'EXPENSES', 'LEGAL_COSTS',
    'MAINTENANCE_AND_PRESERVATION_COSTS', 'TAXES_AND_INSURANCE',
    'MISC_EXPENSES', 'ACTUAL_LOSS_CALCULATION', 'MODIFICATION_COST',
    'STEP_MODIFICATION_FLAG', 'DEFERRED_PAYMENT_PLAN_INDICATOR',
    'ESTIMATED_LOAN_TO_VALUE', 'ZERO_BALANCE_REMOVAL_UPB',
    'DELINQUENCY_DUE_TO_DISASTER', 'BORROWER_ASSISTANCE_STATUS_CODE'
]
```

### 3. Termination & Default Codes

**Termination Codes** (ZERO_BALANCE_CODE):
- `02`: Third Party Sale
- `03`: Short Sale or Short Payoff (DEFAULT)
- `06`: Repurchase
- `09`: REO Disposition (DEFAULT)

**Default Definition**: Codes 03 and 09 only

### 4. Macroeconomic Variables

Additional variables merged from external sources:

**FRED (Federal Reserve Economic Data)**:
- `INTEREST_RATE_30YR`: 30-Year Fixed Rate Mortgage Average
- `DELINQUENCY_RATE_FRED`: Delinquency Rate on Single-Family Residential Mortgages
- `CHARGEOFF_RATE`: Charge-Off Rate on All Loans, All Commercial Banks

**LAUS (Local Area Unemployment Statistics)**:
- `UNEMPLOYMENT_RATE`: County-level unemployment rate
- `UNEMPLOYMENT_AT_ORIGINATION`: Unemployment at loan origination

**FMHPI (Freddie Mac House Price Index)**:
- `HPI`: State-level house price index

**Computed Variables**:
- `ELTV`: Estimated LTV (computed from UPB and HPI)

### Total: **31 Variables** (as per paper Table 1)

## Feature Engineering

### 1. Temporal Variables
```python
df['year'] = df['MONTHLY_REPORTING_PERIOD'] // 100
df['month'] = df['MONTHLY_REPORTING_PERIOD'] % 100
df['LOAN_AGE'] = computed from origination and reporting dates
```

### 2. One-Hot Encoding
Categorical variables encoded:
- `PROPERTY_STATE`
- `CHANNEL`
- `FIRST_TIME_HOMEBUYER`
- `LOAN_PURPOSE`
- `PROPERTY_TYPE`
- `OCCUPANCY_STATUS`

**Result**: 95 features after encoding

### 3. Normalization
All features standardized using `StandardScaler`:
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

## Model Architecture

### LSTM Network

```
Input: (batch_size, sequence_length, 95 features)
    ↓
LSTM Layer 1 (64 hidden units)
    ↓
Dropout (0.2)
    ↓
LSTM Layer 2 (64 hidden units)
    ↓
Dropout (0.2)
    ↓
LSTM Layer 3 (64 hidden units)
    ↓
Dropout (0.2)
    ↓
LSTM Layer 4 (64 hidden units)
    ↓
Dropout (0.2)
    ↓
Fully Connected (64 units)
    ↓
ReLU Activation
    ↓
Dropout (0.2)
    ↓
Fully Connected (1 unit)
    ↓
Sigmoid → Probability
```

**Hyperparameters**:
- Hidden units: 64
- LSTM layers: 4
- Dropout rate: 0.2
- FC layers: 2 (64 → 1)
- Activation: ReLU (hidden), Sigmoid (output)

## Federated Learning

### FedAvg Algorithm

```python
# Pseudocode
for round in 1..T:
    # 1. Server broadcasts global model
    w_global = global_model.parameters()
    
    # 2. Each client k trains locally
    for client_k in clients:
        w_k = local_train(w_global, data_k, epochs=E)
    
    # 3. Server aggregates
    w_global = (1/K) * Σ(w_k)  # Simple average
    
    # 4. Update global model
    global_model.load(w_global)
```

**Parameters**:
- Global rounds (T): 100
- Local epochs (E): 10
- Clients (K): 14 financial institutions
- Aggregation: Unweighted average (FedAvg)

### Client Partitioning

Clients partitioned by `SELLER_NAME`:
```
Client  1: WELLS FARGO BANK, N.A.
Client  2: COUNTRYWIDE HOME LOANS, INC.
Client  3: Other sellers (aggregated)
Client  4: WASHINGTON MUTUAL BANK
...
Client 14: TAYLOR, BEAN & WHITAKER MORTGAGE CORP.
```

## Training Pipeline

### 1. Data Split
```
2006-2007: Training (65%)
2008:      Validation (20%) - for calibration
2009:      Testing (15%)
```

### 2. Optimization
```python
optimizer = SGD(
    parameters,
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001  # L2 regularization
)

loss_fn = BCEWithLogitsLoss()
```

### 3. Training Loop
```python
for epoch in range(local_epochs):
    for batch in dataloader:
        sequences, labels = batch
        
        # Forward pass
        outputs = model(sequences)
        loss = loss_fn(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Calibration Methods

### 1. Platt Scaling
Fits logistic regression on model outputs:
```
p_calibrated = sigmoid(A * logit(p_uncal) + B)
```

### 2. Isotonic Regression
Non-parametric, monotonic mapping:
```
p_calibrated = isotonic_fit(p_uncal)
```

### 3. Temperature Scaling ⭐
Single parameter T that scales logits:
```
p_calibrated = sigmoid(logit(p_uncal) / T)
```

### 4. Beta Calibration
Beta distribution with 3 parameters (a, b, c):
```
p_calibrated = p^a / (p^a + (1-p)^b)
```

## Evaluation Metrics

### Classification Metrics
1. **Accuracy** = (TP + TN) / (TP + FP + TN + FN)
2. **Recall** = TP / (TP + FN)
3. **Precision** = TP / (TP + FP)
4. **F1 Score** = 2*TP / (2*TP + FP + FN)

### Calibration Metrics
1. **ECE** (Expected Calibration Error):
   ```
   ECE = Σ (n_b/n) * |acc(b) - conf(b)|
   ```
   Lower is better (0 = perfect)

2. **Brier Score**:
   ```
   Brier = (1/n) * Σ (p_i - y_i)²
   ```
   Lower is better (0 = perfect)

## Implementation Notes

### Sequence Handling
- Variable-length sequences padded to max length (60 time steps)
- Custom collate function for batching
- Sequences sorted by length for efficient LSTM processing

### Memory Management
- Batch processing to manage GPU memory
- Gradient accumulation for large datasets
- Model checkpointing every 10 rounds

### Reproducibility
- Fixed random seed: 42
- Deterministic CUDA operations
- Same data splits across experiments

## References

1. Lee, J. et al. (2023). "Federated Learning for Credit Risk Assessment"
2. McMahan, H.B. et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data"
3. Guo, C. et al. (2017). "On Calibration of Modern Neural Networks"
