"""
================================================================================
PROJECT CONFIGURATION & SETTINGS
================================================================================
Purpose: Centralized configuration for Fish Farm Disease Outbreak Prediction
Author: AI Assistant
Date: March 2026
Version: 2.0

This module contains:
1. Directory and file path configurations
2. Data features (input features and target variable)
3. Model parameters and hyperparameters
4. Risk thresholds and parameter ranges
5. Streamlit UI configuration
6. Risk assessment messages

Usage:
    import config.settings as settings
    raw_data = pd.read_csv(settings.RAW_DATA_FILE)
    features = settings.INPUT_FEATURES
    
Note: Modify values here to change model behavior without editing source code
================================================================================
"""

import os

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
# Automatically determine project root directory for portability

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_ARTIFACTS_DIR = os.path.join(MODELS_DIR, 'model_artifacts')

# Create directories if they don't exist (safety measure)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_ARTIFACTS_DIR, exist_ok=True)

# ============================================================================
# FILE PATHS
# ============================================================================
# Paths to data files, models, and results

RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, 'fish_farm_dataset.csv')
PROCESSED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'fish_farm_processed.csv')
BEST_MODEL_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'best_model.pkl')
SCALER_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'scaler.pkl')
FEATURE_NAMES_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'feature_names.pkl')
TRAINING_RESULTS_PATH = os.path.join(MODELS_DIR, 'training_results.csv')

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Core ML model parameters

TEST_SIZE = 0.2                 # 80% train, 20% test split
VALIDATION_SIZE = 0.2           # Additional validation set size
RANDOM_STATE = 42               # Random seed for reproducibility
N_SPLITS = 5                    # K-fold cross-validation splits

# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================
# 9 water quality and fish behavior parameters used for prediction

INPUT_FEATURES = [
    'Temperature_C',            # Water temperature (degrees Celsius)
    'pH',                       # pH level (acidity/alkalinity)
    'Dissolved_Oxygen_mg_L',    # Oxygen concentration (mg/L)
    'Ammonia_mg_L',             # Ammonia concentration (mg/L)
    'Nitrate_mg_L',             # Nitrate concentration (mg/L)
    'Turbidity_NTU',            # Water clarity (Nephelometric Turbidity Units)
    'Feed_Intake_Percent',      # Fish feeding rate (%)
    'Growth_Rate_g_week',       # Fish weekly growth (grams)
    'Mortality_Count_per_day'   # Fish deaths per day
]

TARGET_FEATURE = 'Disease_Outbreak'  # Binary: 0=Stable, 1=Risk/Outbreak

# ============================================================================
# WATER PARAMETER VALIDATION RANGES
# ============================================================================
# Min-Max valid ranges for each parameter (acceptable in aquaculture)

WATER_PARAM_RANGES = {
    'Temperature_C': (15, 35),
    'pH': (5.5, 8.5),
    'Dissolved_Oxygen_mg_L': (2, 10),
    'Ammonia_mg_L': (0, 5),
    'Nitrate_mg_L': (0, 100),
    'Turbidity_NTU': (2, 100),
    'Feed_Intake_Percent': (20, 100),
    'Growth_Rate_g_week': (0.2, 3.0),
    'Mortality_Count_per_day': (0, 50)
}

# ============================================================================
# OPTIMAL WATER PARAMETER RANGES
# ============================================================================
# Ideal ranges for healthy fish farm conditions (disease-free environment)

OPTIMAL_RANGES = {
    'Temperature_C': (24, 28),
    'pH': (6.5, 8.0),
    'Dissolved_Oxygen_mg_L': (5, 10),
    'Ammonia_mg_L': (0, 0.5),
    'Nitrate_mg_L': (0, 40),
    'Turbidity_NTU': (2, 5),
    'Feed_Intake_Percent': (80, 100),
    'Growth_Rate_g_week': (1.2, 3.0),
    'Mortality_Count_per_day': (0, 2)
}

# ============================================================================
# RISK THRESHOLDS
# ============================================================================
# Model probability threshold for classification

RISK_THRESHOLD = 0.5           # 0.0-0.5 = Stable (Low Risk)
                               # 0.5-1.0 = Risk (High Risk/Outbreak Possible)

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================
# Tuned parameters for each ML algorithm
LOGISTIC_REGRESSION_PARAMS = {
    'max_iter': 1000,
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced'
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced',
    'n_jobs': -1
}

XGBOOST_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'scale_pos_weight': 1,
    'eval_metric': 'logloss'
}

SVM_PARAMS = {
    'kernel': 'rbf',
    'C': 1.0,
    'gamma': 'scale',
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced'
}

# ============================================================================
# HYPERPARAMETER TUNING GRID
# ============================================================================
RANDOM_FOREST_GRID = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

XGBOOST_GRID = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [5, 6, 7]
}

# ============================================================================
# STREAMLIT CONFIGURATION
# ============================================================================
STREAMLIT_PAGE_CONFIG = {
    'page_title': 'Fish Farm Disease Prediction',
    'page_icon': '🐟',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# ============================================================================
# RISK THRESHOLDS FOR CATEGORIZATION (BINARY CLASSIFICATION)
# ============================================================================
RISK_THRESHOLD = 0.5            # 0.0 - 0.5 = STABLE, 0.5 - 1.0 = RISK

# ============================================================================
# COLORS & STYLING
# ============================================================================
COLOR_STABLE = '#d1fae5'        # Light green background for STABLE
COLOR_RISK = '#fee2e2'          # Light red background for RISK

# ============================================================================
# MESSAGES & RECOMMENDATIONS
# ============================================================================
RISK_MESSAGES = {
    'STABLE': {
        'title': '✅ STABLE',
        'description': 'Water quality is optimal. Farm conditions are healthy.',
        'recommendation': 'Continue regular monitoring. Maintain current management practices.',
        'action_items': [
            '✓ Maintain current feeding schedule',
            '✓ Continue regular water testing',
            '✓ Monitor fish behavior daily',
            '✓ Keep temperature stable'
        ]
    },
    'RISK': {
        'title': '🚨 RISK',
        'description': 'Water parameters indicate potential disease risk. Immediate attention required.',
        'recommendation': 'Implement preventive measures and increase monitoring frequency immediately.',
        'action_items': [
            '⚠️ Increase monitoring frequency to 4-6 times daily',
            '⚠️ Check and adjust water parameters (pH, Oxygen, Ammonia)',
            '⚠️ Consider partial water change (25-30%)',
            '⚠️ Prepare isolation tank for affected fish',
            '⚠️ Consult with aquaculture veterinarian',
            '⚠️ Review and adjust feeding patterns'
        ]
    }
}

# ============================================================================
# PRINT CONFIGURATION HELPER
# ============================================================================
def print_config():
    """Print all configuration values for debugging"""
    print("=" * 70)
    print("PROJECT CONFIGURATION")
    print("=" * 70)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Raw Data File: {RAW_DATA_FILE}")
    print(f"Processed Data File: {PROCESSED_DATA_FILE}")
    print(f"Best Model Path: {BEST_MODEL_PATH}")
    print(f"Scaler Path: {SCALER_PATH}")
    print(f"Feature Names: {INPUT_FEATURES}")
    print("=" * 70)

if __name__ == '__main__':
    print_config()
