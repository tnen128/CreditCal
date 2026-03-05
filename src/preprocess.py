

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib
import json

# Configuration
# Data file path - adjust if your data is in a different location
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "replication_dataset_strict.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "preprocessed_strict"
N_INSTITUTIONS = 14
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

def main():
    print("="*80)
    print("FL CREDIT RISK - DATA PREPROCESSING PIPELINE")
    print("="*80)
    
    # ========================================================================
    # Step 1: Load Data
    # ========================================================================
    print("\n[1/8] Loading data...")
    df = pd.read_csv(DATA_FILE)
    print(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # ========================================================================
    # Step 2: Data Cleaning
    # ========================================================================
    print("\n[2/8] Data cleaning...")
    
    # Drop 100% missing columns
    drop_cols = ['SUPER_CONFORMING_FLAG', 'DELINQUENCY_DUE_TO_DISASTER']
    print(f"  Dropping columns with 100% missingness: {drop_cols}")
    df = df.drop(columns=drop_cols, errors='ignore')
    
    # Forward-fill macroeconomic features within loan sequences
    macro_cols = ['HPI', 'UNEMPLOYMENT_RATE', 'UNEMPLOYMENT_AT_ORIGINATION', 
                  'ELTV', 'DELINQUENCY_RATE_FRED']
    print(f"  Forward-filling macroeconomic features: {macro_cols}")
    for col in macro_cols:
        if col in df.columns:
            df[col] = df.groupby('LOAN_SEQUENCE_NUMBER')[col].ffill()
    
    # Drop any remaining rows with missing values
    rows_before = len(df)
    df = df.dropna()
    rows_after = len(df)
    print(f"  Dropped {rows_before - rows_after:,} rows with remaining NaNs")
    print(f"  Final: {rows_after:,} rows")
    
    # ========================================================================
    # Step 3: Temporal Splitting
    # ========================================================================
    print("\n[3/8] Temporal splitting...")
    
    # Extract year from MONTHLY_REPORTING_PERIOD
    df['year'] = df['MONTHLY_REPORTING_PERIOD'].astype(str).str[:4].astype(int)
    df['month'] = df['MONTHLY_REPORTING_PERIOD'].astype(str).str[4:6].astype(int)
    
    # Split by year
    train_df = df[df['year'].isin([2006, 2007])].copy()
    val_df = df[df['year'] == 2008].copy()
    test_df = df[df['year'] == 2009].copy()
    
    print(f"  Train (2006-2007): {len(train_df):,} rows")
    print(f"  Val (2008): {len(val_df):,} rows")
    print(f"  Test (2009): {len(test_df):,} rows")
    
    # ========================================================================
    # Step 4: Institution Filtering (Top 14)
    # ========================================================================
    print("\n[4/8] Filtering to Top 14 institutions...")
    
    # Count loans per seller
    seller_counts = df.groupby('SELLER_NAME').size().sort_values(ascending=False)
    
    # Exclude aggregated sellers
    excluded_sellers = ['OTHER SELLERS', 'OTHER', 'VARIOUS', 'MULTIPLE', 'UNKNOWN']
    valid_sellers = seller_counts[~seller_counts.index.str.upper().isin(excluded_sellers)]
    
    # Select Top 14
    TOP_14 = valid_sellers.head(N_INSTITUTIONS).index.tolist()
    
    print(f"  Top {N_INSTITUTIONS} institutions:")
    for i, seller in enumerate(TOP_14, 1):
        count = seller_counts[seller]
        default_rate = df[df['SELLER_NAME'] == seller]['DEFAULT_LABEL'].mean() * 100
        print(f"    {i:2d}. {seller[:50]:50s} {count:7,} loans ({default_rate:.2f}% default)")
    
    # Filter all datasets
    train_df = train_df[train_df['SELLER_NAME'].isin(TOP_14)]
    val_df = val_df[val_df['SELLER_NAME'].isin(TOP_14)]
    test_df = test_df[test_df['SELLER_NAME'].isin(TOP_14)]
    
    print(f"\n  After filtering:")
    print(f"    Train: {len(train_df):,} rows")
    print(f"    Val: {len(val_df):,} rows")
    print(f"    Test: {len(test_df):,} rows")
    
    # ========================================================================
    # Step 5: Feature Identification
    # ========================================================================
    print("\n[5/8] Identifying features...")
    
    # Columns to exclude from features
    exclude = ['LOAN_SEQUENCE_NUMBER', 'MONTHLY_REPORTING_PERIOD', 'DEFAULT_LABEL', 
               'year', 'month', 'SELLER_NAME']
    
    # Numerical columns
    numeric_cols = ['CREDIT_SCORE', 'ORIG_INTEREST_RATE', 'ORIG_UPB', 'ORIG_LOAN_TERM', 
                   'LTV', 'MI_PERCENT', 'CURRENT_ACTUAL_UPB', 'LOAN_AGE', 
                   'REMAINING_MONTHS_TO_LEGAL_MATURITY', 'HPI', 'UNEMPLOYMENT_RATE', 
                   'INTEREST_RATE_30YR', 'DELINQUENCY_RATE_FRED', 'ELTV', 'NUM_UNITS', 
                   'NUM_BORROWERS', 'CURRENT_LOAN_DELINQUENCY_STATUS', 'DTI', 
                   'UNEMPLOYMENT_AT_ORIGINATION', 'CHARGEOFF_RATE']
    
    # Ensure all numeric columns exist and are numeric
    for col in numeric_cols:
        if col in train_df.columns:
            for _df in [train_df, val_df, test_df]:
                _df[col] = pd.to_numeric(_df[col], errors='coerce').fillna(0)
    
    # Categorical columns (everything else that's not in exclude)
    potential_features = [c for c in train_df.columns if c not in exclude]
    cat_cols = [c for c in potential_features if c not in numeric_cols]
    
    print(f"  Numerical features: {len(numeric_cols)}")
    print(f"  Categorical features: {len(cat_cols)} - {cat_cols}")
    
    # ========================================================================
    # Step 6: Categorical Encoding
    # ========================================================================
    print("\n[6/8] Encoding categorical features...")
    
    # One-hot encoding
    train_encoded = pd.get_dummies(train_df, columns=cat_cols, dummy_na=True)
    val_encoded = pd.get_dummies(val_df, columns=cat_cols, dummy_na=True)
    test_encoded = pd.get_dummies(test_df, columns=cat_cols, dummy_na=True)
    
    # Align columns across all splits
    feature_cols = [c for c in train_encoded.columns if c not in exclude]
    
    for df_encoded in [val_encoded, test_encoded]:
        # Add missing columns
        for col in feature_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        # Remove extra columns
        extra_cols = [c for c in df_encoded.columns if c not in train_encoded.columns]
        df_encoded.drop(columns=extra_cols, inplace=True, errors='ignore')
    
    print(f"  Total features after encoding: {len(feature_cols)}")
    
    # ========================================================================
    # Step 7: Numerical Scaling
    # ========================================================================
    print("\n[7/8] Scaling numerical features...")
    
    scaler = StandardScaler()
    scaler.fit(train_encoded[feature_cols].fillna(0))
    
    # Note: We don't transform here; transformation happens in DataLoader
    # Save scaler for later use
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scaler_path = OUTPUT_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler fitted and saved: {scaler_path}")
    
    # ========================================================================
    # Step 8: Save Preprocessed Data
    # ========================================================================
    print("\n[8/8] Saving preprocessed data...")
    
    # Save as CSV
    train_encoded.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_encoded.to_csv(OUTPUT_DIR / "val.csv", index=False)
    test_encoded.to_csv(OUTPUT_DIR / "test.csv", index=False)
    
    print(f"  Train saved: {OUTPUT_DIR / 'train.csv'}")
    print(f"  Val saved: {OUTPUT_DIR / 'val.csv'}")
    print(f"  Test saved: {OUTPUT_DIR / 'test.csv'}")
    
    # Save metadata
    metadata = {
        'random_seed': RANDOM_SEED,
        'n_institutions': N_INSTITUTIONS,
        'top_14_institutions': TOP_14,
        'feature_cols': feature_cols,
        'numeric_cols': numeric_cols,
        'categorical_cols': cat_cols,
        'train_size': len(train_encoded),
        'val_size': len(val_encoded),
        'test_size': len(test_encoded),
        'n_features': len(feature_cols),
        'class_distribution': {
            'train': train_encoded['DEFAULT_LABEL'].value_counts().to_dict(),
            'val': val_encoded['DEFAULT_LABEL'].value_counts().to_dict(),
            'test': test_encoded['DEFAULT_LABEL'].value_counts().to_dict()
        }
    }
    
    with open(OUTPUT_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Metadata saved: {OUTPUT_DIR / 'metadata.json'}")
    


if __name__ == "__main__":
    main()
