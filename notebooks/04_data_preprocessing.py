"""
STEP 4: DATA PREPROCESSING
Handle missing values and data cleaning
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.settings as settings

print("=" * 80)
print("STEP 4: DATA PREPROCESSING")
print("=" * 80)

df = pd.read_csv(settings.RAW_DATA_FILE).copy()

print("\n📌 BEFORE PREPROCESSING:")
print(f"Missing values:\n{df.isnull().sum()}\n")

# Handle missing values
print("🔧 HANDLING MISSING VALUES:")

# Numerical columns → fill with mean
numerical_cols = df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mean(), inplace=True)
        print(f"  ✓ Filled {col} with mean value")

# Categorical columns → fill with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)
        print(f"  ✓ Filled {col} with mode value")

print("\n✅ AFTER PREPROCESSING:")
print(f"Missing values:\n{df.isnull().sum()}\n")

# Drop duplicates if any
duplicates = df.duplicated().sum()
if duplicates > 0:
    df = df.drop_duplicates()
    print(f"✓ Removed {duplicates} duplicate rows\n")

print("=" * 80 + "\n")
