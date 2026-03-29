"""
================================================================================
STEP 8: MODEL EVALUATION
================================================================================
Purpose: Evaluate and compare performance of all trained models

Metrics Calculated for Each Model:
1. Accuracy          - Percentage of correct predictions
2. Precision         - True positives / (True positives + False positives)
                      How many predicted outbreaks are correct
3. Recall            - True positives / (True positives + False negatives)
                      What percentage of actual outbreaks were detected
4. F1-Score          - Harmonic mean of Precision and Recall
                      Balanced metric (0.0-1.0 scale)
5. ROC-AUC           - Area under the ROC curve (0.5-1.0 scale)
                      Model's ability to distinguish between classes

Additional Output:
- Confusion Matrix   - 2x2 matrix showing prediction outcomes
- Classification Report - Detailed metrics by class
- Side-by-side comparison table

Key Interpretation:
- Higher metrics generally indicate better model performance
- F1-Score balances precision and recall
- ROC-AUC is useful for imbalanced datasets
- Confusion matrix shows types of errors

Execution:
    python 08_model_evaluation.py
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import xgboost as xgb
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.settings as settings

print("=" * 80)
print("STEP 8: MODEL EVALUATION")
print("=" * 80)

# Prepare data (load, clean, split)
df = pd.read_csv(settings.RAW_DATA_FILE).copy()
df = df.dropna()

X = df[settings.INPUT_FEATURES]
y = df[settings.TARGET_FEATURE]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=settings.TEST_SIZE,
    random_state=settings.RANDOM_STATE,
    stratify=y
)

# Define all models
models = {
    'Logistic Regression': LogisticRegression(random_state=settings.RANDOM_STATE, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=settings.RANDOM_STATE, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=settings.RANDOM_STATE),
    'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=settings.RANDOM_STATE),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=settings.RANDOM_STATE, eval_metric='logloss')
}

results = []

print(f"\n📊 MODEL EVALUATION RESULTS:\n")
print("=" * 100)

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Store results
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    })
    
    # Print detailed results for each model
    print(f"\n{name}")
    print("-" * 100)
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    # Show confusion matrix (2x2 matrix)
    print(f"\n  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"    [{cm[0][0]}, {cm[0][1]}]")
    print(f"    [{cm[1][0]}, {cm[1][1]}]")

print("\n" + "=" * 100)

# Create comparison table of all models
results_df = pd.DataFrame(results)
print("\n📈 MODELS COMPARISON TABLE:")
print(results_df.to_string(index=False))

print("\n" + "=" * 80 + "\n")
