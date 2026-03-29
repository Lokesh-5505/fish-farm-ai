"""
================================================================================
STEP 3: DATA EXPLORATION
================================================================================
Purpose: Analyze dataset structure, statistics, and data types

Key Tasks:
1. Check dataset shape (number of rows and columns)
2. Display column names and data types
3. Generate statistical summary (mean, std, min, max)
4. Show detailed data information
5. Identify missing values

Outputs:
- Data shape dimensions
- Column names and types
- Statistical summary table
- Data info (memory usage, non-null counts)
- Missing value report

Importance:
- Understand data before preprocessing
- Identify data quality issues
- Plan appropriate preprocessing strategies

Execution:
    python 03_data_exploration.py
================================================================================
"""

import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.settings as settings

print("=" * 80)
print("STEP 3: DATA EXPLORATION")
print("=" * 80)

df = pd.read_csv(settings.RAW_DATA_FILE)

# 3A: Data Shape - Check dimensions
print("\n📏 DATA SHAPE:")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(f"Shape output: {df.shape}\n")

# 3B: Column Names - List all features
print("📋 COLUMN NAMES:")
print(df.columns.tolist())
print()

# 3C: Data Types - Check variable types
print("📌 DATA TYPES:")
print(df.dtypes)
print()

# 3D: Statistical Summary - Get descriptive statistics
print("\n📊 STATISTICAL SUMMARY:")
print(df.describe())
print()

# 3E: Data Info - Memory and non-null information
print("\n🔍 DATA INFO:")
df.info()
print()

# 3F: Missing Values - Identify null values
print("\n❌ MISSING VALUES:")
print(df.isnull().sum())
print()

print("=" * 80 + "\n")
