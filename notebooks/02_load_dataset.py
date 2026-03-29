"""
STEP 2: LOAD DATASET
Load and display the Fish Farm Disease dataset
"""

import pandas as pd
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.settings as settings

print("=" * 80)
print("STEP 2: LOAD DATASET")
print("=" * 80)

# Load dataset
data_path = settings.RAW_DATA_FILE
print(f"\n📁 Loading dataset from: {data_path}")

df = pd.read_csv(data_path)

print("\n✓ Dataset loaded successfully!")
print(f"\n📊 DATASET PREVIEW (First 5 rows):\n")
print(df.head())

print("\n" + "=" * 80 + "\n")
