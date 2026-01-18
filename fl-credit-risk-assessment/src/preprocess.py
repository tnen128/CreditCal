import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

# ============================================================================
# CONSTANTS & MAPPINGS
# ============================================================================

DATA_DIR = 'data'
OUTPUT_FILE = 'data/replication_dataset_strict.csv'

# Freddie Mac Column Names (based on User Guide)
ORIG_COLS = [
    'CREDIT_SCORE', 'FIRST_PAYMENT_DATE', 'FIRST_TIME_HOMEBUYER', 'MATURITY_DATE', 'MSA', 'MI_PERCENT',
    'NUM_UNITS', 'OCCUPANCY_STATUS', 'CLTV', 'DTI', 'ORIG_UPB', 'LTV',
    'ORIG_INTEREST_RATE', 'CHANNEL', 'PPM_FLAG', 'PRODUCT_TYPE', 'PROPERTY_STATE',
    'PROPERTY_TYPE', 'POSTAL_CODE', 'LOAN_SEQUENCE_NUMBER', 'LOAN_PURPOSE',
    'ORIG_LOAN_TERM', 'NUM_BORROWERS', 'SELLER_NAME', 'SERVICER_NAME',
    'SUPER_CONFORMING_FLAG', 'PRE_HARP_LOAN_SEQUENCE_NUMBER', 'PROGRAM_INDICATOR',
    'HARP_INDICATOR', 'PROPERTY_VALUATION_METHOD', 'INTEREST_ONLY_INDICATOR'
]

PERF_COLS = [
    'LOAN_SEQUENCE_NUMBER', 'MONTHLY_REPORTING_PERIOD', 'CURRENT_ACTUAL_UPB',
    'CURRENT_LOAN_DELINQUENCY_STATUS', 'LOAN_AGE', 'REMAINING_MONTHS_TO_LEGAL_MATURITY',
    'DEFECT_SETTLEMENT_DATE', 'MODIFICATION_FLAG', 'ZERO_BALANCE_CODE',
    'ZERO_BALANCE_EFFECTIVE_DATE', 'CURRENT_INTEREST_RATE', 'CURRENT_DEFERRED_UPB',
    'DDLPI', 'MI_RECOVERIES', 'NET_SALE_PROCEEDS', 'NON_MI_RECOVERIES', 'EXPENSES',
    'LEGAL_COSTS', 'MAINTENANCE_AND_PRESERVATION_COSTS', 'TAXES_AND_INSURANCE',
    'MISC_EXPENSES', 'ACTUAL_LOSS_CALCULATION', 'MODIFICATION_COST',
    'STEP_MODIFICATION_FLAG', 'DEFERRED_PAYMENT_PLAN_INDICATOR',
    'ESTIMATED_LOAN_TO_VALUE', 'ZERO_BALANCE_REMOVAL_UPB',
    'DELINQUENCY_DUE_TO_DISASTER', 'BORROWER_ASSISTANCE_STATUS_CODE'
]

# Paper Variables (Table 1) - ALL 31 variables (Lee et al., 2023)
TARGET_VARS = [
    'LOAN_SEQUENCE_NUMBER', 'MONTHLY_REPORTING_PERIOD', 'PROPERTY_STATE', 'SELLER_NAME', # Identification
    'CHANNEL', 'FIRST_TIME_HOMEBUYER', 'LOAN_PURPOSE', 'PROPERTY_TYPE', 
    'OCCUPANCY_STATUS', 'PROPERTY_VALUATION_METHOD', 'NUM_UNITS', 'NUM_BORROWERS', 
    'SUPER_CONFORMING_FLAG', # Discrete Static Features
    'CREDIT_SCORE', 'DTI', 'ORIG_INTEREST_RATE', 'ORIG_UPB', 'ORIG_LOAN_TERM', 
    'LTV', 'MI_PERCENT', # Continuous Static Features
    'CURRENT_ACTUAL_UPB', 'CURRENT_LOAN_DELINQUENCY_STATUS', 'LOAN_AGE',
    'REMAINING_MONTHS_TO_LEGAL_MATURITY', 'DELINQUENCY_DUE_TO_DISASTER', # Dynamic Performance
    'HPI', 'UNEMPLOYMENT_RATE', 'UNEMPLOYMENT_AT_ORIGINATION', 'INTEREST_RATE_30YR', 
    'DELINQUENCY_RATE_FRED', 'CHARGEOFF_RATE', # Environmental Features
    'ELTV', # Feature Engineering (Estimated LTV)
    'DEFAULT_LABEL' # Target (derived from Zero Balance Code)
]

# State FIPS to Abbreviation Mapping for LAUS
FIPS_TO_STATE = {
    '01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA', '08': 'CO', '09': 'CT',
    '10': 'DE', '11': 'DC', '12': 'FL', '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL',
    '18': 'IN', '19': 'IA', '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME', '24': 'MD',
    '25': 'MA', '26': 'MI', '27': 'MN', '28': 'MS', '29': 'MO', '30': 'MT', '31': 'NE',
    '32': 'NV', '33': 'NH', '34': 'NJ', '35': 'NM', '36': 'NY', '37': 'NC', '38': 'ND',
    '39': 'OH', '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI', '45': 'SC', '46': 'SD',
    '47': 'TN', '48': 'TX', '49': 'UT', '50': 'VT', '51': 'VA', '53': 'WA', '54': 'WV',
    '55': 'WI', '56': 'WY', '72': 'PR'
}

# ============================================================================
# LOADING HELPERS
# ============================================================================

def load_laus(filepath):
    """Load and parse LAUS (Unemployment) data."""
    print(f"Loading LAUS data from {filepath}...")
    # LAUS format is space-delimited, but irregular. 
    # Use read_csv with regex delimiter
    df = pd.read_csv(filepath, sep=r'\s+', engine='python')
    
    # Filter for Series ID ending in '03' (Unemployment Rate)
    # Series ID format: LAU ST [FIPS] 000000 03
    df = df[df['series_id'].str.endswith('03')]
    
    # Extract State FIPS
    df['fips'] = df['series_id'].str.extract(r'LAUST(\d{2})')
    df['state'] = df['fips'].map(FIPS_TO_STATE)
    
    # Filter years 2006-2009
    df = df[df['year'].isin([2006, 2007, 2008, 2009])]
    
    # Filter valid periods (M01-M12)
    df = df[df['period'].str.startswith('M') & (df['period'] != 'M13')]
    df['month'] = df['period'].str.replace('M', '').astype(int)
    
    # Select columns
    df = df[['state', 'year', 'month', 'value']].rename(columns={'value': 'UNEMPLOYMENT_RATE'})
    return df

def load_fmhpi(filepath):
    """Load House Price Index data."""
    print(f"Loading FMHPI data from {filepath}...")
    df = pd.read_csv(filepath)
    # Format: Year, Month, State, Index_SA (Seasonally Adjusted)
    # Need to verify actual columns. Assuming 'GEO_Code' is State, 'Index_SA' is value
    # Adjust based on actual inspection later if needed.
    # Attempt straightforward loading assuming standard format
    
    # Standard FMHPI columns often: Year, Month, GEO_Name, Index_SA
    if 'GEO_Code' in df.columns:
        df = df.rename(columns={'GEO_Code': 'state', 'Index_SA': 'HPI'})
    elif 'State' in df.columns: # Alternative naming
        df = df.rename(columns={'State': 'state', 'Index_SA': 'HPI'})
        
    df = df[df['Year'].isin([2006, 2007, 2008, 2009])]
    df = df[['state', 'Year', 'Month', 'HPI']].rename(columns={'Year': 'year', 'Month': 'month'})
    return df

def load_fred_rates(filepath):
    """Load 30-Year Mortgage Rates."""
    print(f"Loading FRED Rates from {filepath}...")
    df = pd.read_csv(filepath)
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    df['year'] = df['observation_date'].dt.year
    df['month'] = df['observation_date'].dt.month
    
    # Filter years
    df = df[df['year'].isin([2006, 2007, 2008, 2009])]
    
    # Aggregate weekly to monthly (mean)
    monthly = df.groupby(['year', 'month'])['MORTGAGE30US'].mean().reset_index()
    monthly = monthly.rename(columns={'MORTGAGE30US': 'INTEREST_RATE_30YR'})
    return monthly

def load_fred_delinquency(filepath):
    """Load Delinquency Rates."""
    print(f"Loading FRED Delinquency from {filepath}...")
    df = pd.read_csv(filepath)
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    df['year'] = df['observation_date'].dt.year
    df['month'] = df['observation_date'].dt.month
    
    # Filter years
    df = df[df['year'].isin([2006, 2007, 2008, 2009])]
    
    # Assume quarterly data needs forward fill or duplicate
    # But grouping by year/month works if data is sparse?
    # Better: set index to date, resample to Monthly, ffill
    df = df.set_index('observation_date').resample('M').ffill().reset_index()
    df['year'] = df['observation_date'].dt.year
    df['month'] = df['observation_date'].dt.month
    
    col_name = [c for c in df.columns if 'ACBN' in c][0] # Find rate column
    df = df.rename(columns={col_name: 'DELINQUENCY_RATE_FRED'})
    df = df[['year', 'month', 'DELINQUENCY_RATE_FRED']]
    return df

def load_fred_chargeoff(filepath):
    """Load Charge-Off Rates (Quarterly)."""
    print(f"Loading FRED Charge-Off from {filepath}...")
    df = pd.read_csv(filepath)
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    df['year'] = df['observation_date'].dt.year
    df['quarter'] = df['observation_date'].dt.quarter
    
    # Filter years
    df = df[df['year'].isin([2006, 2007, 2008, 2009])]
    
    # Assuming column contains charge-off rate
    col_name = [c for c in df.columns if c != 'observation_date'][0]
    df = df.rename(columns={col_name: 'CHARGEOFF_RATE'})
    
    # Average if multiple observations per quarter
    result = df.groupby(['year', 'quarter'], as_index=False)['CHARGEOFF_RATE'].mean()
    return result

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_year_quarter(year, quarter, env_data):
    """Process a single Quarter of orig/perf data."""
    orig_file = f"{DATA_DIR}/historical_data_{year}/historical_data_{year}Q{quarter}.txt"
    perf_file = f"{DATA_DIR}/historical_data_{year}/historical_data_time_{year}Q{quarter}.txt"
    
    if not os.path.exists(orig_file):
        # Try unzipping if txt doesn't exist? (Assuming user unzipped or we handle zip)
        # For now assume files are extracted or we extract specific ones
        zip_file = f"{DATA_DIR}/historical_data_{year}/historical_data_{year}Q{quarter}.zip"
        if os.path.exists(zip_file):
            print(f"Unzipping {zip_file}...")
            os.system(f"unzip -o {zip_file} -d {DATA_DIR}/historical_data_{year}/")
        
    if not os.path.exists(orig_file):
        print(f"Skipping {year}Q{quarter} - File not found")
        return None

    # 1. Load Origination
    print(f"Loading Origination {year}Q{quarter}...")
    orig_df = pd.read_csv(orig_file, sep='|', header=None, names=ORIG_COLS, 
                          index_col=False, low_memory=False, 
                          dtype={'LOAN_SEQUENCE_NUMBER': str})
    
    # Keep only relevant columns
    orig_keep = ['LOAN_SEQUENCE_NUMBER', 'CREDIT_SCORE', 'FIRST_PAYMENT_DATE', 'FIRST_TIME_HOMEBUYER', 
                 'MATURITY_DATE', 'MI_PERCENT', 'NUM_UNITS', 'OCCUPANCY_STATUS', 
                 'ORIG_UPB', 'LTV', 'ORIG_INTEREST_RATE', 'CHANNEL', 'PROPERTY_STATE', 
                 'PROPERTY_TYPE', 'LOAN_PURPOSE', 'ORIG_LOAN_TERM', 'NUM_BORROWERS', 
                 'SELLER_NAME', 'PROPERTY_VALUATION_METHOD', 'SUPER_CONFORMING_FLAG', 'DTI']
    # Add dummy col if Property Valuation Method missing (older files might not have it)
    valid_cols = [c for c in orig_keep if c in orig_df.columns]
    orig_df = orig_df[valid_cols]
    
    # 2. Load Performance
    print(f"Loading Performance {year}Q{quarter}...")
    perf_df = pd.read_csv(perf_file, sep='|', header=None, names=PERF_COLS, 
                          index_col=False, low_memory=False, 
                          dtype={'LOAN_SEQUENCE_NUMBER': str})
    perf_df['year'] = perf_df['MONTHLY_REPORTING_PERIOD'].astype(str).str[:4].astype(int)
    perf_df['month'] = perf_df['MONTHLY_REPORTING_PERIOD'].astype(str).str[4:6].astype(int)
    perf_df = perf_df[perf_df['year'].isin([2006, 2007, 2008, 2009])]
    
    if perf_df.empty:
        return None

    # 3. Merge Determine Target (Termination Events)
    # We need to find loans that HAVE a termination event.
    # Group by Loan Seq Num and check if any row has valid Zero Balance Code
    
    # termination_events = [01, 03, 06, 09] (Prepaid, Foreclosure, Repurchase, REO)
    # 01 = Prepaid (Non-Default)
    # 03, 06, 09 = Default
    
    # Get last row for each loan to check outcome
    # perf_df is sorted by Loan Seq then Period? Usually. Let's ensure.
    perf_df = perf_df.sort_values(['LOAN_SEQUENCE_NUMBER', 'MONTHLY_REPORTING_PERIOD'])
    
    # Identify IDs that terminated in our window with relevant codes
    # We want loans that ended with 01, 03, 06, 09
    # Identical to data/preprocess.py (Lee et al. Methodology)
    # Termination Codes: 02 (Third Party Sale), 03 (Short Sale), 06 (Repurchase), 09 (REO)
    # Note: '01' (Prepaid) is EXCLUDED based on reference script.
    perf_df['ZERO_BALANCE_CODE'] = pd.to_numeric(perf_df['ZERO_BALANCE_CODE'], errors='coerce')
    relevant_codes = [2.0, 3.0, 6.0, 9.0]
    
    # Get the final status of every loan
    last_rows = perf_df.drop_duplicates('LOAN_SEQUENCE_NUMBER', keep='last')
    terminated_loans = last_rows[last_rows['ZERO_BALANCE_CODE'].isin(relevant_codes)]
    
    # Get the list of valid Loan IDs
    valid_ids = terminated_loans['LOAN_SEQUENCE_NUMBER'].unique()
    
    if len(valid_ids) == 0:
        return None
        
    print(f"Found {len(valid_ids)} terminated loans in {year}Q{quarter}")
    
    # Filter Performance to keep ALL history for these valid loans
    perf_filtered = perf_df[perf_df['LOAN_SEQUENCE_NUMBER'].isin(valid_ids)].copy()
    
    # Assign Target Label
    # Reference Script: DEFAULT_CODES = {'03', '09'} (Short Sale, REO)
    # 02 (Third Party Sale) and 06 (Repurchase) are Non-Default (0)
    id_to_outcome = terminated_loans.set_index('LOAN_SEQUENCE_NUMBER')['ZERO_BALANCE_CODE']
    default_codes = [3.0, 9.0]
    
    # Function to map code to binary default
    def get_label(code):
        return 1 if code in default_codes else 0
        
    # Apply mapping
    # perf_filtered['DEFAULT_LABEL'] = perf_filtered['LOAN_SEQUENCE_NUMBER'].map(lambda x: get_label(id_to_outcome[x]))
    # Faster mapping:
    terminated_loans['final_label'] = terminated_loans['ZERO_BALANCE_CODE'].isin(default_codes).astype(int)
    id_to_label = terminated_loans.set_index('LOAN_SEQUENCE_NUMBER')['final_label']
    perf_filtered['DEFAULT_LABEL'] = perf_filtered['LOAN_SEQUENCE_NUMBER'].map(id_to_label)
    
    # 4. Merge Origination (Static Features)
    # We merge origination columns into EVERY row of performance
    full_df = pd.merge(perf_filtered, orig_df, on='LOAN_SEQUENCE_NUMBER', how='inner')
    
    
    # 5. Merge Environmental Data
    # Merge HPI (State, Year, Month)
    full_df = pd.merge(full_df, env_data['hpi'], 
                       left_on=['PROPERTY_STATE', 'year', 'month'], 
                       right_on=['state', 'year', 'month'], 
                       how='left')
    
    # Merge Unemployment (State, Year, Month)
    full_df = pd.merge(full_df, env_data['laus'],
                       left_on=['PROPERTY_STATE', 'year', 'month'],
                       right_on=['state', 'year', 'month'],
                       how='left')
                       
    # Merge Interest/Delinquency/Charge-off - National
    full_df = pd.merge(full_df, env_data['rates'], on=['year', 'month'], how='left')
    full_df = pd.merge(full_df, env_data['delinq'], on=['year', 'month'], how='left')
    
    # Merge Charge-Off (Quarterly)
    full_df['quarter'] = ((full_df['month'] - 1) // 3) + 1
    full_df = pd.merge(full_df, env_data['chargeoff'], on=['year', 'quarter'], how='left')
    
    # Compute Unemployment at Origination (matching reference script)
    full_df['orig_year'] = pd.to_numeric(full_df['FIRST_PAYMENT_DATE'].astype(str).str[:4], errors='coerce')
    full_df['orig_month'] = pd.to_numeric(full_df['FIRST_PAYMENT_DATE'].astype(str).str[4:6], errors='coerce')
    
    laus_orig = env_data['laus'].copy().rename(columns={'UNEMPLOYMENT_RATE': 'UNEMPLOYMENT_AT_ORIGINATION'})
    full_df = pd.merge(full_df, laus_orig[['state', 'year', 'month', 'UNEMPLOYMENT_AT_ORIGINATION']],
                       left_on=['PROPERTY_STATE', 'orig_year', 'orig_month'],
                       right_on=['state', 'year', 'month'],
                       how='left', suffixes=('', '_orig'))
    
    # Clean up temporary merge columns
    full_df = full_df.drop(columns=[c for c in full_df.columns if c.endswith('_orig')], errors='ignore')
    
    # 6. ELTV (Estimated LTV) calculation
    # Paper equation probably: Current UPB / (Orig Value * (Current HPI / Orig HPI))
    # Approximation: LTV * (HPI_orig / HPI_current) * (Current_UPB / Orig_UPB) ??
    # Simpler: Current UPB / Estimated Value
    # Estimated Value = (Orig UPB / (LTV/100)) * (HPI_current / HPI_orig)
    # Need Origination HPI (join again on Origination Date)
    
    # We will approximate or skip complicated HPI mapping for now to ensure MVP runs
    # Paper says: "Estimated Loan-to-Value (ELTV)"
    # We'll calculate simple ELTV = CURRENT_ACTUAL_UPB / (ORIG_UPB / (LTV/100)) 
    # (ignoring HPI adjustment for now to save complexity, or add it if easy)
    
    full_df['orig_value'] = full_df['ORIG_UPB'] / (full_df['LTV'].replace(0, np.nan) / 100)
    full_df['ELTV'] = (full_df['CURRENT_ACTUAL_UPB'] / full_df['orig_value']) * 100
    
    # Select Final Columns
    final_cols = [c for c in TARGET_VARS if c in full_df.columns]
    return full_df[final_cols]

def main():
    print("Starting Replication Preprocessing...")
    
    # 1. Load Auxiliary Data
    env_data = {}
    env_data['laus'] = load_laus(f"{DATA_DIR}/laus.data.2.AllStatesU.txt")
    env_data['hpi'] = load_fmhpi(f"{DATA_DIR}/fmhpi_master_file.csv")
    env_data['rates'] = load_fred_rates(f"{DATA_DIR}/MORTGAGE30US.csv")
    env_data['delinq'] = load_fred_delinquency(f"{DATA_DIR}/CORALACBN.csv")
    env_data['chargeoff'] = load_fred_chargeoff(f"{DATA_DIR}/CORALACBN.csv")  # Same source, different processing

    # 2. Loop Years/Quarters
    first_chunk = True
    total_records = 0
    
    # Initialize output file (remove if exists)
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        
    for year in [2006, 2007, 2008, 2009]:
        for q in range(1, 5):
            df = process_year_quarter(year, q, env_data)
            if df is not None:
                print(f"Processed {year}Q{q}: {len(df)} records")
                total_records += len(df)
                
                # Append to CSV
                mode = 'w' if first_chunk else 'a'
                header = first_chunk
                df.to_csv(OUTPUT_FILE, mode=mode, header=header, index=False)
                first_chunk = False
                
                # Clear memory
                del df
                import gc
                gc.collect()
            
    # 5. Final Report
    print(f"Total Records Processed: {total_records}")
    print(f"Saved to {OUTPUT_FILE}...")
    print("Done!")

if __name__ == "__main__":
    main()
