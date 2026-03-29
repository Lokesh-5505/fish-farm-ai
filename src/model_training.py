"""
================================================================================
MODEL TRAINING MODULE
================================================================================
Purpose: Train multiple ML models for fish farm disease outbreak prediction
Author: AI Assistant
Date: March 2026
Version: 2.0

Key Functions:
- run_faculty_format_pipeline(): Comprehensive 13-step ML workflow
- Trains 5 different models: Logistic Regression, Random Forest, SVM, XGBoost, etc.
- Performs model evaluation, comparison, and selection
- Saves best model artifacts for deployment

Workflow Steps:
1. Problem Understanding
2. Library imports
3. Data Loading
4. Data Exploration
5. Data Cleaning (handle missing values)
6. EDA (Exploratory Data Analysis)
7. Feature Engineering
8. Data Splitting (80% train, 20% test)
9. Model Training (multiple algorithms)
10. Model Evaluation (metrics, confusion matrix)
11. Best Model Selection (by F1-Score)
12. Model Saving (joblib serialization)
13. Results Export (training results CSV)
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report)
import joblib
import pickle
import sys
import os

# Import project configuration and settings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.settings as settings


def _print_step(step_no, title):
    """
    Print a standard section header for workflow steps
    
    Args:
        step_no (int): Step number in the workflow
        title (str): Title/name of the step
    """
    print("\n" + "-" * 60)
    print(f"Step {step_no}: {title}")
    print("-" * 60)

def run_faculty_format_pipeline():
    """
    Execute complete 13-step machine learning workflow for disease outbreak prediction
    
    This function orchestrates the entire ML pipeline:
    - Problem definition and understanding
    - Library imports and setup
    - Data loading and exploration
    - Data cleaning and preprocessing
    - Feature engineering and scaling
    - Train-test splitting
    - Model training (multiple algorithms)
    - Model evaluation and comparison
    - Best model selection
    - Model and artifacts saving
    - Results export
    
    Returns:
        None (saves models and results to disk)
    """
    # ===== STEP 1: Problem Understanding =====
    # Define the problem: Binary classification for disease outbreak prediction
    _print_step(1, "Problem Understanding")
    print("Type: Binary Classification")
    print("Objective: Predict disease outbreak risk (0 = No, 1 = Yes)")
    print(f"Input Features ({len(settings.INPUT_FEATURES)}): {', '.join(settings.INPUT_FEATURES)}")
    print(f"Target: {settings.TARGET_FEATURE}")

    _print_step(2, "Import Libraries")
    print("Libraries imported: pandas, numpy, scikit-learn, xgboost, joblib")

    _print_step(3, "Load Data")
    df = pd.read_csv(settings.RAW_DATA_FILE)
    print(f"Source: {settings.RAW_DATA_FILE}")
    print("\nFirst 5 records:")
    print(df.head())

    _print_step(4, "Data Understanding")
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {', '.join(df.columns.tolist())}")
    print("\nData types:")
    print(df.dtypes)
    print("\nDescriptive statistics:")
    print(df.describe(include='all'))

    _print_step(5, "Data Cleaning")
    print("Missing values (before):")
    print(df.isnull().sum())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    for col in categorical_cols:
        mode_series = df[col].mode(dropna=True)
        if len(mode_series) > 0:
            df[col] = df[col].fillna(mode_series.iloc[0])

    print("\nMissing values (after):")
    print(df.isnull().sum())

    _print_step(6, "Exploratory Data Analysis")
    print("Class distribution:")
    print(df[settings.TARGET_FEATURE].value_counts())

    feature_means = df[settings.INPUT_FEATURES].mean().sort_values(ascending=False)
    print("\nTop 3 features by mean:")
    print(feature_means.head(3))

    correlations = df[settings.INPUT_FEATURES + [settings.TARGET_FEATURE]].corr()[settings.TARGET_FEATURE].drop(settings.TARGET_FEATURE)
    print("\nFeature correlation with target (top 3):")
    print(correlations.abs().sort_values(ascending=False).head(3))

    _print_step(7, "Feature Engineering")
    required_cols = settings.INPUT_FEATURES + [settings.TARGET_FEATURE]
    irrelevant_cols = [col for col in df.columns if col not in required_cols]

    if irrelevant_cols:
        df = df.drop(columns=irrelevant_cols)
        print(f"Removed: {irrelevant_cols}")
    else:
        print("No irrelevant columns.")

    label_encoders = {}
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in categorical_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        label_encoders[col] = encoder

    if categorical_cols:
        print(f"Categorical encoding: {categorical_cols}")

    X = df[settings.INPUT_FEATURES].copy()
    y = df[settings.TARGET_FEATURE].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Feature scaling: StandardScaler applied")

    _print_step(8, "Train-Test Split")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=settings.TEST_SIZE,
        random_state=settings.RANDOM_STATE,
        stratify=y
    )
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    _print_step(9, "Model Training")
    lr_model = LogisticRegression(max_iter=1000, random_state=settings.RANDOM_STATE)
    lr_model.fit(X_train, y_train)
    print("Logistic Regression: trained")

    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=settings.RANDOM_STATE,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    print("Random Forest: trained")

    _print_step(10, "Model Evaluation")
    lr_pred = lr_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)

    lr_accuracy = accuracy_score(y_test, lr_pred)
    rf_accuracy = accuracy_score(y_test, rf_pred)

    print("Logistic Regression:")
    print(f"  Accuracy: {lr_accuracy:.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test, lr_pred)}")
    lr_metrics = {
        'Precision': precision_score(y_test, lr_pred, zero_division=0),
        'Recall': recall_score(y_test, lr_pred, zero_division=0),
        'F1-Score': f1_score(y_test, lr_pred, zero_division=0)
    }
    print(f"  {lr_metrics}")

    print("\nRandom Forest:")
    print(f"  Accuracy: {rf_accuracy:.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test, rf_pred)}")
    rf_metrics = {
        'Precision': precision_score(y_test, rf_pred, zero_division=0),
        'Recall': recall_score(y_test, rf_pred, zero_division=0),
        'F1-Score': f1_score(y_test, rf_pred, zero_division=0)
    }
    print(f"  {rf_metrics}")

    _print_step(11, "Model Comparison")
    if rf_accuracy >= lr_accuracy:
        best_model_name = 'Random Forest'
        best_model = rf_model
    else:
        best_model_name = 'Logistic Regression'
        best_model = lr_model

    print(f"Best model: {best_model_name}")
    print(f"Test accuracy: {max(lr_accuracy, rf_accuracy):.4f}")

    _print_step(12, "Model Persistence")
    faculty_model_path = os.path.join(settings.MODEL_ARTIFACTS_DIR, 'faculty_best_model.pkl')
    faculty_scaler_path = os.path.join(settings.MODEL_ARTIFACTS_DIR, 'faculty_standard_scaler.pkl')

    with open(faculty_model_path, 'wb') as model_file:
        pickle.dump(best_model, model_file)
    with open(faculty_scaler_path, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    print(f"Model saved: {faculty_model_path}")
    print(f"Scaler saved: {faculty_scaler_path}")

    _print_step(13, "Sample Prediction")
    sample_input = X.iloc[[0]].copy()
    sample_scaled = scaler.transform(sample_input)
    sample_prediction = best_model.predict(sample_scaled)[0]

    if hasattr(best_model, 'predict_proba'):
        sample_proba = best_model.predict_proba(sample_scaled)[0][1]
        print(f"Predicted class: {sample_prediction}")
        print(f"Confidence (P(class=1)): {sample_proba:.4f}")
    else:
        print(f"Predicted class: {sample_prediction}")

    print("\nWorkflow completed.")


class ModelTrainer:
    """Train and evaluate multiple ML models"""
    
    def __init__(self, random_state=settings.RANDOM_STATE):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
    
    def load_and_prepare_data(self, filepath):
        """Load data and split into train/test sets"""
        print("\n" + "-"*60)
        print("DATA LOADING AND PREPARATION")
        print("-"*60)
        
        # Load data
        data = pd.read_csv(filepath)
        print(f"Dataset loaded: {data.shape}")
        
        # Separate features and target
        X = data[settings.INPUT_FEATURES].copy()
        y = data[settings.TARGET_FEATURE].copy()
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=settings.TEST_SIZE,
            random_state=self.random_state,
            stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Class distribution (train): {np.bincount(self.y_train)}")
        print(f"Class distribution (test): {np.bincount(self.y_test)}")
        
        # Save scaler
        joblib.dump(self.scaler, settings.SCALER_PATH)
        print(f"Scaler saved: {settings.SCALER_PATH}")
    
    def train_logistic_regression(self):
        """Train Logistic Regression model"""
        print("\n" + "-"*60)
        print("Training: Logistic Regression")
        print("-"*60)
        
        model = LogisticRegression(**settings.LOGISTIC_REGRESSION_PARAMS)
        model.fit(self.X_train, self.y_train)
        
        self.models['Logistic Regression'] = model
        print("Training completed")
        
        return model
    
    def train_random_forest(self, tune=False):
        """Train Random Forest model"""
        print("\n" + "-"*60)
        print("Training: Random Forest")
        print("-"*60)
        
        if tune:
            print("Hyperparameter tuning in progress...")
            rf = RandomForestClassifier(random_state=self.random_state)
            
            grid_search = GridSearchCV(
                rf,
                settings.RANDOM_FOREST_GRID,
                cv=5,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(self.X_train, self.y_train)
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            
            model = grid_search.best_estimator_
        else:
            model = RandomForestClassifier(**settings.RANDOM_FOREST_PARAMS)
            model.fit(self.X_train, self.y_train)
        
        self.models['Random Forest'] = model
        print("Training completed")
        
        return model
    
    def train_xgboost(self, tune=False):
        """Train XGBoost model"""
        print("\n" + "-"*60)
        print("Training: XGBoost")
        print("-"*60)
        
        if tune:
            print("Hyperparameter tuning in progress...")
            xgb = XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss'
            )
            
            grid_search = GridSearchCV(
                xgb,
                settings.XGBOOST_GRID,
                cv=5,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(self.X_train, self.y_train)
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            
            model = grid_search.best_estimator_
        else:
            model = XGBClassifier(**settings.XGBOOST_PARAMS)
            model.fit(self.X_train, self.y_train)
        
        self.models['XGBoost'] = model
        print("Training completed")
        
        return model
    
    def train_svm(self, tune=False):
        """Train Support Vector Machine model"""
        print("\n" + "-"*60)
        print("Training: Support Vector Machine")
        print("-"*60)
        
        model = SVC(**settings.SVM_PARAMS, probability=True)
        model.fit(self.X_train, self.y_train)
        
        self.models['SVM'] = model
        print("Training completed")
        
        return model
    
    def evaluate_model(self, model_name, model):
        """Evaluate a single model"""
        # Make predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        # Store results
        self.results[model_name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        }
    
    def evaluate_all_models(self):
        """Evaluate all trained models"""
        print("\n" + "-"*60)
        print("MODEL EVALUATION")
        print("-"*60 + "\n")
        
        results_summary = []
        
        for model_name, model in self.models.items():
            print(f"Evaluating {model_name}...")
            metrics = self.evaluate_model(model_name, model)
            
            results_summary.append({
                'Model': model_name,
                'Accuracy': metrics['Accuracy'],
                'Precision': metrics['Precision'],
                'Recall': metrics['Recall'],
                'F1-Score': metrics['F1-Score'],
                'ROC-AUC': metrics['ROC-AUC']
            })
            
            print(f"  Accuracy: {metrics['Accuracy']:.4f}")
            print(f"  Precision: {metrics['Precision']:.4f}")
            print(f"  Recall: {metrics['Recall']:.4f}")
            print(f"  F1-Score: {metrics['F1-Score']:.4f}")
            print(f"  ROC-AUC: {metrics['ROC-AUC']:.4f}")
        
        # Create results dataframe
        results_df = pd.DataFrame(results_summary)
        results_df = results_df.sort_values('F1-Score', ascending=False)
        
        print("\n" + "-"*60)
        print("MODEL COMPARISON")
        print("-"*60)
        print(results_df.to_string(index=False))
        
        # Save results
        results_df.to_csv(settings.TRAINING_RESULTS_PATH, index=False)
        print(f"\nResults saved: {settings.TRAINING_RESULTS_PATH}")
        
        return results_df
    
    def select_best_model(self):
        """Select best model based on F1-Score"""
        best_f1 = -1
        
        for model_name, metrics in self.results.items():
            # Calculate F1 if not in results
            if 'F1-Score' not in metrics:
                self.evaluate_model(model_name, self.models[model_name])
            
            f1 = self.results[model_name]['F1-Score']
            
            if f1 > best_f1:
                best_f1 = f1
                self.best_model_name = model_name
                self.best_model = self.models[model_name]
        
        print("\n" + "-"*60)
        print(f"BEST MODEL SELECTED: {self.best_model_name}")
        print("-"*60)
        print(f"F1-Score: {best_f1:.4f}")
        
        return self.best_model
    
    def save_best_model(self):
        """Save best model to disk"""
        if self.best_model is not None:
            joblib.dump(self.best_model, settings.BEST_MODEL_PATH)
            print(f"Model saved: {settings.BEST_MODEL_PATH}")
            
            # Also save feature names for consistency
            joblib.dump(settings.INPUT_FEATURES, settings.FEATURE_NAMES_PATH)
            print(f"Features saved: {settings.FEATURE_NAMES_PATH}")
        else:
            print("Error: No best model available")
    
    def train_all_models(self, tune_hyperparams=False):
        """Train all models in sequence"""
        print("\n" + "-"*60)
        print("MODEL TRAINING PHASE")
        print("-"*60)
        
        self.train_logistic_regression()
        self.train_random_forest(tune=tune_hyperparams)
        self.train_xgboost(tune=tune_hyperparams)
        self.train_svm()
        
        print("\nAll models training completed.")



if __name__ == '__main__':
    if '--faculty-format' in sys.argv:
        print("Fish Farm Disease Prediction - Educational Workflow\n")
        run_faculty_format_pipeline()
    else:
        print("Fish Farm Disease Prediction - Model Training\n")

        # Initialize trainer
        trainer = ModelTrainer()

        # Load and prepare data
        trainer.load_and_prepare_data(settings.RAW_DATA_FILE)

        # Train all models
        trainer.train_all_models(tune_hyperparams=False)

        # Evaluate all models
        results_df = trainer.evaluate_all_models()

        # Select and save best model
        trainer.select_best_model()
        trainer.save_best_model()

        print("\nTraining pipeline completed.")
        print(f"Best model: {trainer.best_model_name}")
