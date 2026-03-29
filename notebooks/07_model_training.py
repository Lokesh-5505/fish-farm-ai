"""
================================================================================
STEP 7: MODEL TRAINING
================================================================================
Purpose: Train multiple machine learning models on prepared data

Models Trained:
1. Logistic Regression   - Linear model for baseline comparison
2. Random Forest         - Ensemble method using decision trees
3. Gradient Boosting     - Sequential boosting for improved accuracy
4. Support Vector Machine - Non-linear kernel-based method
5. XGBoost               - Advanced gradient boosting framework

Process:
- All models trained on same preprocessed data
- Uses 80% training data, 20% test data
- Stratified split maintains class balance
- Each model stores predictions for evaluation

Outputs:
- Trained model objects in memory
- Ready for evaluation and comparison
- Basis for best model selection

Training Parameters:
- Random State: 42 (reproducibility)
- Max Iterations: 1000 (convergence)
- N Estimators: 100 (tree count for ensemble)

Execution:
    python 07_model_training.py
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.settings as settings

print("=" * 80)
print("STEP 7: MODEL TRAINING")
print("=" * 80)

# Prepare data (load, clean, scale)
df = pd.read_csv(settings.RAW_DATA_FILE).copy()
df = df.dropna()

# Extract features and target
X = df[settings.INPUT_FEATURES]
y = df[settings.TARGET_FEATURE]

# Scale features to 0-1 range
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=settings.TEST_SIZE,
    random_state=settings.RANDOM_STATE,
    stratify=y
)

print(f"\n🤖 TRAINING MODELS:\n")

# Model 1: Logistic Regression - Linear baseline
print("  1️⃣  Training Logistic Regression...")
lr = LogisticRegression(random_state=settings.RANDOM_STATE, max_iter=1000)
lr.fit(X_train, y_train)
print("     ✓ Completed")

# Model 2: Random Forest - Ensemble of decision trees
print("  2️⃣  Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=settings.RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)
print("     ✓ Completed")

# Model 3: Gradient Boosting - Sequential tree boosting
print("  3️⃣  Training Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=100, random_state=settings.RANDOM_STATE)
gb.fit(X_train, y_train)
print("     ✓ Completed")

# Model 4: Support Vector Machine - Kernel-based classifier
print("  4️⃣  Training Support Vector Machine...")
svm = SVC(kernel='rbf', probability=True, random_state=settings.RANDOM_STATE)
svm.fit(X_train, y_train)
print("     ✓ Completed")

# Model 5: XGBoost - Advanced gradient boosting (typically best performance)
print("  5️⃣  Training XGBoost...")
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=settings.RANDOM_STATE, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
print("     ✓ Completed")

print("\n✅ ALL MODELS TRAINED SUCCESSFULLY!")
print("\n" + "=" * 80 + "\n")

# Store models for next step
models = {
    'Logistic Regression': lr,
    'Random Forest': rf,
    'Gradient Boosting': gb,
    'Support Vector Machine': svm,
    'XGBoost': xgb_model
}
