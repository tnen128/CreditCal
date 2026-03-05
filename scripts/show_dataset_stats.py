#!/usr/bin/env python3
import sys
sys.path.append('src')
import json
import pandas as pd

# Load metadata
with open('data/preprocessed_strict/metadata.json') as f:
    metadata = json.load(f)

print('=' * 70)
print('DATASET: Fannie Mae Single-Family Loan Performance Data')
print('=' * 70)
print(f'Time Period: 2006-2009 (Financial Crisis)')
print(f'Total Institutions: {len(metadata["top_14_institutions"])}')
print()

# Load data to count
train = pd.read_csv('data/preprocessed_strict/train.csv')
val = pd.read_csv('data/preprocessed_strict/val.csv')
test = pd.read_csv('data/preprocessed_strict/test.csv')

print('Institution Breakdown (by unique loans):')
print('-' * 70)
print(f"{'Institution':<45} {'Train':<8} {'Val':<7} {'Test':<7} {'Total':<8}")
print('-' * 70)

for inst in metadata['top_14_institutions']:
    train_loans = train[train['SELLER_NAME'] == inst]['LOAN_SEQUENCE_NUMBER'].nunique()
    val_loans = val[val['SELLER_NAME'] == inst]['LOAN_SEQUENCE_NUMBER'].nunique()
    test_loans = test[test['SELLER_NAME'] == inst]['LOAN_SEQUENCE_NUMBER'].nunique()
    total = train_loans + val_loans + test_loans
    print(f'{inst:<45} {train_loans:<8} {val_loans:<7} {test_loans:<7} {total:<8}')

print('-' * 70)
total_train = train['LOAN_SEQUENCE_NUMBER'].nunique()
total_val = val['LOAN_SEQUENCE_NUMBER'].nunique()
total_test = test['LOAN_SEQUENCE_NUMBER'].nunique()
grand_total = total_train + total_val + total_test
print(f"{'TOTAL':<45} {total_train:<8} {total_val:<7} {total_test:<7} {grand_total:<8}")
print()
print(f'Note: Train = 2006-2007, Val = 2008, Test = 2009')
print()

# Show default rates
print('Default Rates by Institution (Test Set 2009):')
print('-' * 70)
print(f"{'Institution':<45} {'Defaults':<10} {'Total':<8} {'Rate':<8}")
print('-' * 70)

for inst in metadata['top_14_institutions']:
    inst_test = test[test['SELLER_NAME'] == inst]
    if len(inst_test) > 0:
        defaults = inst_test['DEFAULT_LABEL'].sum()
        total = len(inst_test['LOAN_SEQUENCE_NUMBER'].unique())
        rate = (defaults / total * 100) if total > 0 else 0
        print(f'{inst:<45} {int(defaults):<10} {total:<8} {rate:>6.2f}%')

total_defaults = test['DEFAULT_LABEL'].sum()
total_test_loans = test['LOAN_SEQUENCE_NUMBER'].nunique()
overall_rate = (total_defaults / total_test_loans * 100)
print('-' * 70)
print(f"{'OVERALL':<45} {int(total_defaults):<10} {total_test_loans:<8} {overall_rate:>6.2f}%")
