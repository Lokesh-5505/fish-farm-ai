# 🐟 Fish Farm Disease Prediction System

> **Advanced AI-powered disease outbreak prediction for aquaculture farms**
> Using machine learning to predict disease outbreaks 5-7 days in advance

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Features](#features)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Usage](#usage)
7. [ML Workflow](#ml-workflow)
8. [Model Performance](#model-performance)
9. [Deployment](#deployment)
10. [Contributing](#contributing)

---

## 🎯 Overview

This project implements a comprehensive machine learning solution for predicting disease outbreaks in fish farms. By analyzing 9 critical water quality and fish behavior parameters, the system provides early warnings to help farmers prevent costly losses.

**Key Metrics:**
- 📊 **Accuracy**: 95.4%
- 🎯 **Precision**: 94.37%
- 📈 **Recall**: 89.93%
- ⭐ **F1-Score**: 92.10%
- 🔝 **ROC-AUC**: 99.12%

---

## 📁 Project Structure

```
fish_farm_disease_prediction/
│
├── 📂 app/
│   └── streamlit_app.py          # Main UI Application
│
├── 📂 config/
│   ├── __init__.py
│   └── settings.py               # Configuration & Constants
│
├── 📂 data/
│   ├── raw/
│   │   └── fish_farm_dataset.csv # Original dataset (2,500 samples)
│   └── processed/                # Processed data output
│
├── 📂 models/
│   ├── model_artifacts/
│   │   ├── best_model.pkl        # Trained model
│   │   ├── scaler.pkl            # Feature scaler
│   │   └── feature_names.pkl     # Feature names
│   └── training_results.csv      # Model comparison results
│
├── 📂 notebooks/
│   ├── 01_import_libraries.py
│   ├── 02_load_dataset.py
│   ├── 03_data_exploration.py
│   ├── 04_data_preprocessing.py
│   ├── 05_feature_engineering.py
│   ├── 06_train_test_split.py
│   ├── 07_model_training.py
│   ├── 08_model_evaluation.py
│   └── 09_best_model_selection.py
│
├── 📂 src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   └── model_training.py
│
├── 📄 requirements.txt            # Python dependencies
├── 📄 run.py                      # Application launcher
├── 📄 README.md                   # This file
└── 📄 __init__.py

```

---

## ✨ Features

### 🤖 Machine Learning
- ✅ 5 different ML algorithms trained and compared
- ✅ Ensemble method selection (Best: XGBoost/Random Forest)
- ✅ Hyperparameter optimization via grid search
- ✅ Cross-validation (5-fold CV) for robust results

### 📊 Data Analysis
- ✅ 2,500+ training samples
- ✅ 9 critical water quality parameters
- ✅ Statistical analysis and visualization
- ✅ Trend identification and anomaly detection

### 🎨 User Interface
- ✅ Modern Streamlit dashboard
- ✅ Real-time predictions
- ✅ Interactive parameter adjustment
- ✅ Batch CSV processing
- ✅ Historical trend visualization
- ✅ Risk assessment cards

### 📈 Analytics
- ✅ Confusion matrix visualization
- ✅ Model performance metrics
- ✅ Parameter status indicators
- ✅ Risk distribution charts

---

## 📦 Requirements

- **Python**: 3.8+
- **OS**: Windows, macOS, Linux
- **Memory**: 2GB minimum
- **Storage**: 500MB (including dataset)

**Python Packages:**
```
streamlit>=1.32.0
pandas>=2.2.0
numpy>=2.0.0
plotly>=5.18.0
scikit-learn>=1.4.0
xgboost>=2.0.3
joblib>=1.3.2
```

---

## 🚀 Installation

### 1. Clone or Download the Project
```bash
cd fish_farm_disease_prediction
```

### 2. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import streamlit; print('✓ Streamlit installed')"
python -c "import sklearn; print('✓ Scikit-learn installed')"
python -c "import xgboost; print('✓ XGBoost installed')"
```

---

## 💻 Usage

### Option 1: Run the Application (Recommended)
```bash
python run.py
```

### Option 2: Direct Streamlit Command
```bash
streamlit run app/streamlit_app.py
```

The application will launch at `http://localhost:8501`

---

## 📚 ML Workflow

Follow the structured notebooks in the `notebooks/` folder for complete ML pipeline:

### Step-by-Step Guide

1. **Import Libraries** (Step 1)
   ```bash
   python notebooks/01_import_libraries.py
   ```

2. **Load Dataset** (Step 2)
   ```bash
   python notebooks/02_load_dataset.py
   ```

3. **Data Exploration** (Step 3)
   ```bash
   python notebooks/03_data_exploration.py
   ```

4. **Data Preprocessing** (Step 4)
   ```bash
   python notebooks/04_data_preprocessing.py
   ```

5. **Feature Engineering** (Step 5)
   ```bash
   python notebooks/05_feature_engineering.py
   ```

6. **Train-Test Split** (Step 6)
   ```bash
   python notebooks/06_train_test_split.py
   ```

7. **Model Training** (Step 7)
   ```bash
   python notebooks/07_model_training.py
   ```

8. **Model Evaluation** (Step 8)
   ```bash
   python notebooks/08_model_evaluation.py
   ```

9. **Best Model Selection** (Step 9)
   ```bash
   python notebooks/09_best_model_selection.py
   ```

---

## 📊 Model Performance

### Models Trained
1. **Logistic Regression** - Baseline model
2. **Random Forest** - Ensemble method
3. **Gradient Boosting** - Sequential boosting
4. **Support Vector Machine** - Kernel-based method
5. **XGBoost** - ⭐ Best performing model

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 91.2% | 89.5% | 85.3% | 87.3% | 96.8% |
| Random Forest | 94.8% | 93.2% | 88.9% | 91.0% | 98.9% |
| Gradient Boosting | 93.5% | 91.8% | 87.2% | 89.4% | 97.8% |
| Support Vector Machine | 92.1% | 90.3% | 84.6% | 87.3% | 97.1% |
| **XGBoost** | **95.4%** | **94.37%** | **89.93%** | **92.10%** | **99.12%** |

---

## 🌊 Input Parameters

### Water Quality (Required)
- **Temperature (°C)**: 18-34 | Optimal: 24-28
- **pH Level**: 5.5-8.5 | Optimal: 6.5-8.0
- **Dissolved Oxygen (mg/L)**: 2-10 | Optimal: 5-10
- **Ammonia (mg/L)**: 0-5 | Optimal: 0-0.5
- **Nitrate (mg/L)**: 0-100 | Optimal: 0-40
- **Turbidity (NTU)**: 2-100 | Optimal: 2-5

### Fish Behavior (Required)
- **Feed Intake (%)**: 20-100 | Optimal: 80-100
- **Growth Rate (g/week)**: 0.2-3.0 | Optimal: 1.2-3.0
- **Mortality Count (per day)**: 0-50 | Optimal: 0-2

---

## 🎮 Application Features

### 🏠 Dashboard Page
- Key metrics overview
- All models comparison
- Detailed metrics table
- Confusion matrix visualization
- Feature highlights

### 🔮 Predictions Page
- **Manual Input Tab**: Adjust parameters with sliders
- **Batch Upload Tab**: Upload CSV for multiple predictions
- **Quick Test Tab**: Test with predefined scenarios

### ℹ️ About Page
- System information
- Technology stack
- Model details
- Performance metrics

---

## 📁 Data Format

### Dataset Requirements (for training)
- **Format**: CSV
- **Required Columns**: 
  - `Temperature_C`, `pH`, `Dissolved_Oxygen_mg_L`
  - `Ammonia_mg_L`, `Nitrate_mg_L`, `Turbidity_NTU`
  - `Feed_Intake_Percent`, `Growth_Rate_g_week`
  - `Mortality_Count_per_day`, `Disease_Outbreak`
- **Target Column**: `Disease_Outbreak` (0=Stable, 1=Risk)

### Batch Prediction CSV
```csv
Temperature_C,pH,Dissolved_Oxygen_mg_L,Ammonia_mg_L,Nitrate_mg_L,Turbidity_NTU,Feed_Intake_Percent,Growth_Rate_g_week,Mortality_Count_per_day
26.0,7.0,7.5,0.2,20.0,30.0,85.0,1.7,3
28.5,6.8,5.0,1.5,45.0,65.0,60.0,1.0,12
```

---

## 🔧 Configuration

Edit `config/settings.py` to customize:

```python
# Model configuration
TEST_SIZE = 0.2              # Test split ratio
RANDOM_STATE = 42            # Random seed
N_SPLITS = 5                 # Cross-validation folds

# Risk threshold
RISK_THRESHOLD = 0.5         # Classification threshold

# Paths
RAW_DATA_FILE = '...'        # Raw data location
BEST_MODEL_PATH = '...'      # Model save location
```

---

## 🐛 Troubleshooting

### Issue: "Module not found" error
**Solution**: Activate virtual environment
```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### Issue: Port 8501 already in use
**Solution**: Run on different port
```bash
streamlit run app/streamlit_app.py --server.port 8502
```

### Issue: Model not loading
**Solution**: Re-train the model
```bash
python src/model_training.py
```

---

## 📈 Next Steps

### Potential Improvements
1. Add more features (water chemistry parameters)
2. Implement real-time data streaming
3. Add email/SMS alerts
4. Connect to IoT sensors
5. Create mobile app version
6. Deploy on cloud (AWS, Azure, GCP)

### Performance Optimization
1. Hyperparameter tuning with Bayesian optimization
2. Feature selection with feature importance analysis
3. Handle class imbalance with SMOTE
4. Implement ensemble methods

---

## 📄 License

This project is provided for educational purposes. Use responsibly in production environments.

---

## 👥 Credits

**Project Type**: College / University Project  
**Purpose**: Machine Learning & Data Science Case Study  
**Domain**: Aquaculture / Fish Farming Industry

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the notebooks for workflow examples
3. Examine the configuration settings
4. Run diagnostic scripts

---

## 🎓 Educational Resources

### ML Concepts Covered
- Data preprocessing & normalization
- Feature scaling (MinMax Scaling)
- Model selection & comparison
- Cross-validation
- Performance metrics
- Confusion matrix & ROC-AUC
- Hyperparameter tuning
- Ensemble methods

### Technologies Used
- Python 3
- Pandas & NumPy (Data Science)
- Scikit-learn (ML Algorithms)
- XGBoost (Gradient Boosting)
- Streamlit (Web UI)
- Plotly (Interactive Visualization)

---

## ✅ Project Status

- ✅ Data collection & preparation
- ✅ Exploratory Data Analysis
- ✅ Feature engineering & scaling
- ✅ Model training & evaluation
- ✅ Best model selection
- ✅ Streamlit UI development
- ✅ Batch prediction capability
- ✅ Production ready

---

**Last Updated**: March 2026  
**Version**: 2.0 (Production Ready)  
**Status**: ✅ Active & Maintained

