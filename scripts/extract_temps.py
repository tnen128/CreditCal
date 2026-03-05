#!/usr/bin/env python3
import sys
sys.path.append('src')
import joblib
from pathlib import Path

print("Temperature Values from Saved Calibrators:")
print("=" * 60)

# Central model - Wells Fargo
cal_path = Path('results/trial_run_complete/Central/WELLS_FARGO_BANK,_N.A./calibrators/temperature.pkl')
if cal_path.exists():
    cal = joblib.load(cal_path)
    print(f"Central - Wells Fargo: T = {cal.temperature:.4f}")

# Central model - Countrywide
cal_path2 = Path('results/trial_run_complete/Central/COUNTRYWIDE_HOME_LOANS,_INC./calibrators/temperature.pkl')
if cal_path2.exists():
    cal2 = joblib.load(cal_path2)
    print(f"Central - Countrywide:  T = {cal2.temperature:.4f}")

# Central model - Chase
cal_path3 = Path('results/trial_run_complete/Central/CHASE_HOME_FINANCE_LLC/calibrators/temperature.pkl')
if cal_path3.exists():
    cal3 = joblib.load(cal_path3)
    print(f"Central - Chase:        T = {cal3.temperature:.4f}")

print("\n(T > 1 means 'cooling down' overconfident predictions)")
print("(T < 1 means 'heating up' underconfident predictions)")
