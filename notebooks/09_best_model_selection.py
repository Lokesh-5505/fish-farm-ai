"""
================================================================================
STEP 9: BEST MODEL SELECTION & DEPLOYMENT
================================================================================
Purpose: Select best performing model and save artifacts for production use

Selection Criteria:
- Primary metric: F1-Score (balances precision and recall)
- Why F1-Score? Useful when both false positives and false negatives matter
- Best model typically has F1-Score > 0.90 for this task

Artifacts Saved:
1. best_model.pkl         - Trained model object (uses joblib serialization)
2. scaler.pkl             - Feature scaler for preprocessing new data
3. feature_names.pkl      - List of input features in correct order
4. training_results.csv   - CSV with all models' metrics

Production Workflow:
1. Load scaler → normalize input data
2. Load model → make predictions
3. Use saved feature names → ensure correct feature order

Benefits of Saving Artifacts:
- Fast deployment (no retraining needed)
- Consistent preprocessing (same scaler)
- Reproducible results
- Easy versioning

Outputs:
- ✓ Best model identified and displayed
- ✓ Performance metrics of best model
- ✓ All 5 model comparison table
- ✓ Artifacts saved to disk
- ✓ Ready for Streamlit deployment

Execution:
    python 09_best_model_selection.py
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.settings as settings

print("=" * 80)
print("STEP 9: BEST MODEL SELECTION & DEPLOYMENT")
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

print(f"\n🔍 EVALUATING ALL MODELS:\n")

# Train and evaluate all models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1,
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
    })

results_df = pd.DataFrame(results)

# Select best model by F1-Score
best_idx = results_df['F1-Score'].idxmax()
best_model_name = results_df.loc[best_idx, 'Model']
best_model = models[best_model_name]

# Display best model information
print(f"🏆 BEST MODEL SELECTED: {best_model_name}")
print(f"   F1-Score: {results_df.loc[best_idx, 'F1-Score']:.4f}")
print(f"   Accuracy: {results_df.loc[best_idx, 'Accuracy']:.4f}")
print(f"   Precision: {results_df.loc[best_idx, 'Precision']:.4f}")
print(f"   Recall: {results_df.loc[best_idx, 'Recall']:.4f}")
print(f"   ROC-AUC: {results_df.loc[best_idx, 'ROC-AUC']:.4f}")

# Re-train best model on full training data for production
best_model.fit(X_train, y_train)

# Save artifacts for deployment
print(f"\n💾 SAVING MODEL & ARTIFACTS:")
os.makedirs(settings.MODEL_ARTIFACTS_DIR, exist_ok=True)

joblib.dump(best_model, settings.BEST_MODEL_PATH)
joblib.dump(scaler, settings.SCALER_PATH)
joblib.dump(settings.INPUT_FEATURES, settings.FEATURE_NAMES_PATH)

print(f"   ✓ Model saved: {settings.BEST_MODEL_PATH}")
print(f"   ✓ Scaler saved: {settings.SCALER_PATH}")
print(f"   ✓ Feature names saved: {settings.FEATURE_NAMES_PATH}")

# Save all model comparison results
results_df.to_csv(settings.TRAINING_RESULTS_PATH, index=False)
print(f"   ✓ Results saved: {settings.TRAINING_RESULTS_PATH}")

print("\n" + "=" * 80)
print("✅ DEPLOYMENT READY!")
print("=" * 80 + "\n")
