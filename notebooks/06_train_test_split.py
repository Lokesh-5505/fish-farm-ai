"""
STEP 6: TRAIN-TEST SPLIT
Prepare data for model training
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.settings as settings

print("=" * 80)
print("STEP 6: TRAIN-TEST SPLIT")
print("=" * 80)

df = pd.read_csv(settings.RAW_DATA_FILE).copy()
df = df.dropna()

# Prepare features and target
X = df[settings.INPUT_FEATURES]
y = df[settings.TARGET_FEATURE]

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=settings.TEST_SIZE,
    random_state=settings.RANDOM_STATE,
    stratify=y
)

print(f"\n📊 DATASET SPLIT:")
print(f"  Total samples: {len(X)}")
print(f"  Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

print(f"\n🏷️ TARGET DISTRIBUTION:")
print(f"  Training set - Class 0: {(y_train == 0).sum()}, Class 1: {(y_train == 1).sum()}")
print(f"  Test set - Class 0: {(y_test == 0).sum()}, Class 1: {(y_test == 1).sum()}")

print(f"\n📏 FEATURE DIMENSIONS:")
print(f"  X_train shape: {X_train.shape}")
print(f"  X_test shape: {X_test.shape}")
print(f"  y_train shape: {y_train.shape}")
print(f"  y_test shape: {y_test.shape}")

print("\n" + "=" * 80 + "\n")
