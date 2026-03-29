"""
STEP 5: FEATURE ENGINEERING
Encoding, scaling, and feature preparation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.settings as settings

print("=" * 80)
print("STEP 5: FEATURE ENGINEERING (Encoding & Scaling)")
print("=" * 80)

df = pd.read_csv(settings.RAW_DATA_FILE).copy()

# Remove missing values
df = df.dropna()

print("\n🔤 LABEL ENCODING (Categorical Variables):")
label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    if col != settings.TARGET_FEATURE:  # Don't encode target separately here
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"  ✓ Encoded column: {col}")

if len(label_encoders) == 0:
    print("  ✓ No categorical columns found (all numeric)")

print("\n📊 FEATURE SCALING (MinMaxScaler):")
scaler = MinMaxScaler()

# Select features to scale (exclude target)
features = settings.INPUT_FEATURES
if all(f in df.columns for f in features):
    X = df[features]
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features)
    print(f"  ✓ Scaled {len(features)} features to [0, 1] range")
    print(f"\n📌 SCALED DATA PREVIEW (First 5 rows):\n{X_scaled_df.head()}")
else:
    print("  ⚠️ Some features not found in dataset")

print("\n" + "=" * 80 + "\n")
