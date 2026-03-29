"""
================================================================================
DATA PREPROCESSING MODULE
================================================================================
Purpose: Data loading, cleaning, and preparation for machine learning models
Author: AI Assistant
Date: March 2026
Version: 2.0

Key Features:
- Load raw data from CSV files
- Handle missing values (mean, median, forward/backward fill)
- Remove duplicate records
- Detect and handle outliers
- Feature scaling and normalization
- Data validation and quality checks

Class: DataPreprocessor
- Encapsulates all preprocessing operations
- Uses sklearn StandardScaler for normalization
- Provides error handling and logging

Usage:
    preprocessor = DataPreprocessor()
    preprocessor.load_data('data.csv')
    preprocessor.handle_missing_values(method='mean')
    preprocessor.remove_duplicates()
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import sys
import os

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.settings as settings


class DataPreprocessor:
    """
    Handles all data preprocessing tasks for disease prediction model
    
    This class provides methods to:
    - Load raw CSV data
    - Validate data integrity
    - Handle missing values
    - Detect and manage duplicates
    - Scale and normalize features
    - Export processed data
    
    Attributes:
        scaler (StandardScaler): Scikit-learn scaler for feature normalization
        data (pd.DataFrame): Current dataset being processed
        processed_data (pd.DataFrame): Output after preprocessing steps
    """
    
    def __init__(self):
        """Initialize DataPreprocessor with scaler and empty data containers"""
        self.scaler = StandardScaler()
        self.data = None
        self.processed_data = None
        
    def load_data(self, filepath):
        """
        Load raw data from CSV file with validation
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded data or None if error occurs
            
        Raises:
            Exception: Prints error message if file not found or invalid
        """
        try:
            self.data = pd.read_csv(filepath)
            print(f"✓ Data loaded successfully!")
            print(f"  Shape: {self.data.shape}")
            print(f"  Columns: {list(self.data.columns)}")
            return self.data
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return None
    
    def check_missing_values(self):
        """
        Check and report missing values in the dataset
        
        Returns:
            bool: True if no missing values, False otherwise
        """
        missing = self.data.isnull().sum()
        missing_pct = (missing / len(self.data)) * 100
        
        if missing.sum() == 0:
            print("✓ No missing values detected!")
            return True
        else:
            print("⚠️ Missing values found:")
            print(missing[missing > 0])
            return False
    
    def handle_missing_values(self, method='mean'):
        """
        Handle missing values using specified strategy
        
        Args:
            method (str): Strategy to use - 'mean', 'median', 'forward_fill', 'backward_fill'
                         Default: 'mean' (best for continuous numerical data)
        """
        if method == 'mean':
            # Fill numeric columns with column mean
            self.data.fillna(self.data.mean(numeric_only=True), inplace=True)
        elif method == 'median':
            # Fill numeric columns with column median (more robust to outliers)
            self.data.fillna(self.data.median(numeric_only=True), inplace=True)
        elif method == 'forward_fill':
            # Forward fill: use previous value (good for time series)
            self.data = self.data.ffill()
        elif method == 'backward_fill':
            # Backward fill: use next value
            self.data = self.data.bfill()
        
        print(f"✓ Missing values handled using '{method}' method")
    
    def remove_duplicates(self):
        """
        Remove duplicate rows from the dataset
        
        Returns:
            pd.DataFrame: Data after duplicate removal
        """
        before = len(self.data)
        self.data = self.data.drop_duplicates()
        after = len(self.data)
        removed = before - after
        
        if removed > 0:
            print(f"✓ Removed {removed} duplicate rows")
        else:
            print("✓ No duplicate rows found")
        
        return self.data
    
    def handle_outliers_iqr(self, columns=None, multiplier=1.5):
        """
        Handle outliers using Interquartile Range (IQR) method
        multiplier: 1.5 (default), higher = more tolerance for outliers
        """
        if columns is None:
            columns = settings.INPUT_FEATURES
        
        outliers_count = 0
        
        for col in columns:
            if col in self.data.columns and self.data[col].dtype in ['float64', 'int64']:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - (multiplier * IQR)
                upper_bound = Q3 + (multiplier * IQR)
                
                outlier_mask = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
                outliers_in_col = outlier_mask.sum()
                
                if outliers_in_col > 0:
                    # Cap outliers to bounds
                    self.data[col] = self.data[col].clip(lower_bound, upper_bound)
                    outliers_count += outliers_in_col
        
        print(f"✓ Handled {outliers_count} outlier values using IQR method")
    
    def validate_ranges(self):
        """Validate that all values are within expected ranges"""
        issues = []
        
        for feature, (min_val, max_val) in settings.WATER_PARAM_RANGES.items():
            if feature in self.data.columns:
                below_min = (self.data[feature] < min_val).sum()
                above_max = (self.data[feature] > max_val).sum()
                
                if below_min > 0 or above_max > 0:
                    issues.append(f"{feature}: {below_min} below min, {above_max} above max")
        
        if issues:
            print("⚠️ Values outside expected ranges:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✓ All values within expected ranges")
    
    def print_statistics(self):
        """Print descriptive statistics"""
        print("\n" + "="*80)
        print("DATASET STATISTICS")
        print("="*80)
        print(self.data[settings.INPUT_FEATURES + [settings.TARGET_FEATURE]].describe())
        print("\n" + "="*80)
        print("CLASS DISTRIBUTION")
        print("="*80)
        print(self.data[settings.TARGET_FEATURE].value_counts())
        print(f"Outbreak Rate: {self.data[settings.TARGET_FEATURE].mean()*100:.2f}%")
    
    def preprocess(self, filepath, handle_outliers=True):
        """
        Complete preprocessing pipeline
        """
        print("\n" + "="*80)
        print("STARTING DATA PREPROCESSING")
        print("="*80 + "\n")
        
        # Load data
        self.load_data(filepath)
        
        # Check missing values
        self.check_missing_values()
        self.handle_missing_values(method='mean')
        
        # Remove duplicates
        self.remove_duplicates()
        
        # Handle outliers
        if handle_outliers:
            self.handle_outliers_iqr(multiplier=1.5)
        
        # Validate ranges
        self.validate_ranges()
        
        # Print statistics
        self.print_statistics()
        
        print("\n✓ Preprocessing completed successfully!")
        print("="*80 + "\n")
        
        return self.data
    
    def separate_features_target(self):
        """Separate features and target variable"""
        X = self.data[settings.INPUT_FEATURES].copy()
        y = self.data[settings.TARGET_FEATURE].copy()
        
        print(f"\n✓ Features (X) shape: {X.shape}")
        print(f"✓ Target (y) shape: {y.shape}")
        
        return X, y
    
    def scale_features(self, X_train, X_test=None, fit=True):
        """
        Scale features using StandardScaler
        """
        if fit:
            X_train_scaled = self.scaler.fit_transform(X_train)
            print("✓ Scaler fitted on training data")
            
            # Save scaler for later use
            joblib.dump(self.scaler, settings.SCALER_PATH)
            print(f"✓ Scaler saved to {settings.SCALER_PATH}")
        else:
            X_train_scaled = self.scaler.transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def save_processed_data(self, filepath=None):
        """Save processed data to CSV"""
        if filepath is None:
            filepath = settings.PROCESSED_DATA_FILE
        
        self.data.to_csv(filepath, index=False)
        print(f"✓ Processed data saved to {filepath}")


# ============================================================================
# STANDALONE FUNCTIONS FOR QUICK USE
# ============================================================================

def quick_load_and_explore(filepath):
    """Quick function to load and explore data"""
    preprocessor = DataPreprocessor()
    data = preprocessor.load_data(filepath)
    
    print("\n" + "="*80)
    print("DATA OVERVIEW")
    print("="*80)
    print(f"\nDataset shape: {data.shape}")
    print(f"\nFirst 5 rows:\n{data.head()}")
    print(f"\nData types:\n{data.dtypes}")
    print(f"\nMissing values:\n{data.isnull().sum()}")
    print(f"\nBasic statistics:\n{data.describe()}")
    
    return preprocessor, data


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("Fish Farm Disease Prediction - Data Preprocessing Module\n")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Run full preprocessing pipeline
    data = preprocessor.preprocess(
        filepath=settings.RAW_DATA_FILE,
        handle_outliers=True
    )
    
    # Separate features and target
    X, y = preprocessor.separate_features_target()
    
    # Save processed data
    preprocessor.save_processed_data()
    
    print("\n✓ All preprocessing steps completed!")
    print(f"  Input features shape: {X.shape}")
    print(f"  Target distribution:\n{y.value_counts()}")
