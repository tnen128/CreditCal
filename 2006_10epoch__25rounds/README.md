# 🧠 Replication Knowledge Base: Delgado Fernandez et al. (2023)

## 1. Project Objective
Replicate the "Federated Learning for Credit Risk Assessment" experiment EXACTLY as described in the paper.

## 2. Dataset Specifications
([Same as previous, omitted for brevity])

## 3. How to Run

### Step 1: Data Preprocessing (Running Now)
The script `preprocess_replication.py` ingests raw data from `data/`, merges it with environmental variables (LAUS, FRED), and outputs `data/replication_dataset.csv`.
```bash
python preprocess_replication.py
```
*Note: This takes significant time as it processes ~16 Quarters of data.*

### Step 2: Verification (Optional)
Check the generated dataset statistics:
```bash
python verify_replication_data.py
```

### Step 3: Federated Learning Experiment
Run the main replication script. This will:
1. Load `data/replication_dataset.csv`.
2. Partition data into 14 Clients (Financial Institutions).
3. Train a Global LSTM Model for 100 Rounds (Federated Averaging).
4. Evaluate Accuracy/Rec/Prec/F1 every round.
```bash
python main_replication.py
```

## 4. Model Architecture (CRITICAL)
**Type**: LSTM-based Neural Network (Implemented in `models_replication.py`)
- **Input**: 31 features (monthly history).
- **Layers**: 4x LSTM (64 hidden) -> Dense(64) -> Dropout -> Dense(1).

## 5. Experiment Settings
- **Global Rounds**: 100
- **Local Epochs**: 10
- **Batch Size**: 128
- **Learning Rate**: 0.01 (SGD)
