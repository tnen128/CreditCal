import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class ReplicationDataset(Dataset):
    """
    Dataset for Delgado Fernandez et al. (2023) Replication.
    
    Handles:
    - Grouping monthly records into sequences by Loan ID.
    - Sorting by month/time.
    - Extracting Static + Dynamic features.
    - Padding sequences.
    
    Target:
    - Binary Default Label (1 if terminated as Default, 0 if Prepaid).
    - Note: Label is the same for all time steps of a loan (Statis classification).
    """
    def __init__(self, csv_file=None, dataframe=None, max_seq_len=60, features=None):
        """
        Args:
            csv_file: Path to preprocessed CSV (Long Format).
            dataframe: Pre-loaded DataFrame (optional, overrides csv_file).
            max_seq_len: Max timesteps.
            features: List of feature names to use.
        """
        if dataframe is not None:
            self.df = dataframe.copy()
        elif csv_file is not None:
            print(f"Loading dataset from {csv_file}...")
            self.df = pd.read_csv(csv_file)
        else:
            raise ValueError("Must provide either csv_file or dataframe")
            
        self.max_seq_len = max_seq_len
        
        # Sort by ID and Time (Year, Month)
        # Assuming we have 'year' and 'month' or 'MONTHLY_REPORTING_PERIOD'
        if 'year' not in self.df.columns:
            # Reconstruct year/month if needed
            self.df['year'] = self.df['MONTHLY_REPORTING_PERIOD'].astype(str).str[:4].astype(int)
            self.df['month'] = self.df['MONTHLY_REPORTING_PERIOD'].astype(str).str[4:6].astype(int)
            
        self.df = self.df.sort_values(['LOAN_SEQUENCE_NUMBER', 'year', 'month'])
        
        if features is None:
            # Exclude non-features
            # SELLER_NAME is excluded because it's used for partitioning and is constant per client in FL
            exclude = ['LOAN_SEQUENCE_NUMBER', 'MONTHLY_REPORTING_PERIOD', 'DEFAULT_LABEL', 
                       'ZERO_BALANCE_CODE', 'year', 'month', 'final_label', 'SELLER_NAME']
            # Also exclude any other metadata if present
            potential_features = [c for c in self.df.columns if c not in exclude]
        else:
            potential_features = features
            
        # SAFETY FIX: Force known numeric columns to be numeric
        # This prevents "mixed types" (e.g. "6.5" vs 6.5) from being detected as object/categorical
        numeric_force = [
            'CREDIT_SCORE', 'ORIG_INTEREST_RATE', 'ORIG_UPB', 'ORIG_LOAN_TERM', 'LTV', 
            'MI_PERCENT', 'CURRENT_ACTUAL_UPB', 'LOAN_AGE', 'REMAINING_MONTHS_TO_LEGAL_MATURITY', 
            'HPI', 'UNEMPLOYMENT_RATE', 'INTEREST_RATE_30YR', 'DELINQUENCY_RATE_FRED', 'ELTV',
            'NUM_UNITS', 'NUM_BORROWERS'
        ]
        
        for col in numeric_force:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)

        # Handle Delinquency Status separately (often mixed '0', '1', 'R', 'XX')
        # We treat it as numeric (months delinquent). Non-numeric (R, XX) -> 0 or -1? 
        # For simplicity in this replication, we coerce to 0 (Current)
        if 'CURRENT_LOAN_DELINQUENCY_STATUS' in self.df.columns:
             self.df['CURRENT_LOAN_DELINQUENCY_STATUS'] = pd.to_numeric(self.df['CURRENT_LOAN_DELINQUENCY_STATUS'], errors='coerce').fillna(0)

        # One-Hot Encode Categorical Variables
        # Identify object columns in potential_features
        cat_cols = self.df[potential_features].select_dtypes(include=['object']).columns.tolist()
        if cat_cols:
            print(f"Encoding categorical columns: {cat_cols}")
            self.df = pd.get_dummies(self.df, columns=cat_cols, dummy_na=True) # Handle NaNs as category often useful
            
            # Update feature list to include new columns (and remove old cat cols)
            # Filter columns that start with cat_cols prefixes
            new_features = [c for c in self.df.columns if c not in exclude and c not in cat_cols]
            # Ensure we only keep numeric now
            self.features = new_features
        else:
            self.features = potential_features
            
        print(f"Final Features ({len(self.features)})")
        
        # Normalize Features
        # Note: In FL, normalization should be per-client or using global stats.
        self.scaler = StandardScaler()
        # Ensure all features are float
        self.df[self.features] = self.df[self.features].astype(float)
        self.df[self.features] = self.scaler.fit_transform(self.df[self.features].fillna(0))
        
        # Group by Loan ID
        print("Grouping by Loan ID...")
        self.grouped = self.df.groupby('LOAN_SEQUENCE_NUMBER')
        self.loan_ids = list(self.grouped.groups.keys())
        print(f"Found {len(self.loan_ids)} unique loans.")
        
    def __len__(self):
        return len(self.loan_ids)
    
    def __getitem__(self, idx):
        loan_id = self.loan_ids[idx]
        group = self.grouped.get_group(loan_id)
        
        # Get dimensions
        seq_len = len(group)
        
        # Get features
        X = group[self.features].values
        
        # Get target (taken from last row, should be consistent)
        y = group['DEFAULT_LABEL'].iloc[-1]
        
        # Truncate or Pad
        if seq_len > self.max_seq_len:
            # Truncate (Keep last N months? Or first? Usually last leading up to event)
            X = X[-self.max_seq_len:]
            seq_len = self.max_seq_len
            
        # Convert to Tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        # Pad if needed (Pre-padding or Post-padding? PyTorch pack_padded works with Post)
        # We will return the sequences unpadded in a collate_fn ideally, 
        # but for simple Dataset, we can return tensors, and DataLoader collate_fn will pad.
        # Here we just return the variable length tensor.
        
        return X_tensor, y_tensor, seq_len

# Custom Collate Function for DataLoader
def collate_fn(batch):
    # Sort batch by sequence length (descending) for pack_padded_sequence
    batch.sort(key=lambda x: x[2], reverse=True)
    
    sequences, targets, lengths = zip(*batch)
    
    # Pad sequences
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    
    targets = torch.stack(targets).view(-1, 1)
    lengths = torch.tensor(lengths)
    
    return sequences_padded, targets, lengths
