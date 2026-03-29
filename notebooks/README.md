# 📚 Machine Learning Workflow Notebooks

This folder contains the complete ML workflow for the Fish Farm Disease Prediction system, organized following the standard Data Science / Machine Learning process.

---

## 📖 Notebook Structure

Each notebook represents a specific step in the ML pipeline:

### 1️⃣ **01_import_libraries.py**
**Step**: Import all required libraries  
**Purpose**: Load dependencies for data processing and modeling  
**Output**: Library import validation

```bash
python 01_import_libraries.py
```

---

### 2️⃣ **02_load_dataset.py**
**Step**: Load dataset and display preview  
**Purpose**: Read raw data from CSV  
**Output**: First 5 rows of dataset

```bash
python 02_load_dataset.py
```

---

### 3️⃣ **03_data_exploration.py**
**Step**: Explore dataset structure and statistics  
**Outputs**:
- Data shape
- Column names
- Data types
- Statistical summary
- Data info
- Missing values

```bash
python 03_data_exploration.py
```

---

### 4️⃣ **04_data_preprocessing.py**
**Step**: Handle missing values and clean data  
**Actions**:
- Fill numerical missing values → mean
- Fill categorical missing values → mode
- Remove duplicates

```bash
python 04_data_preprocessing.py
```

---

### 5️⃣ **05_feature_engineering.py**
**Step**: Encoding & Feature Scaling  
**Actions**:
- Label encode categorical variables
- Apply MinMaxScaler [0, 1] normalization

```bash
python 05_feature_engineering.py
```

---

### 6️⃣ **06_train_test_split.py**
**Step**: Split data for training and testing  
**Configuration**:
- Train/Test ratio: 80/20
- Stratified split (maintain class balance)
- Random seed: 42

```bash
python 06_train_test_split.py
```

---

### 7️⃣ **07_model_training.py**
**Step**: Train multiple ML models  
**Models Trained**:
1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. Support Vector Machine
5. XGBoost

```bash
python 07_model_training.py
```

---

### 8️⃣ **08_model_evaluation.py**
**Step**: Evaluate and compare all models  
**Metrics Calculated**:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

```bash
python 08_model_evaluation.py
```

---

### 9️⃣ **09_best_model_selection.py**
**Step**: Select best model and save artifacts  
**Actions**:
- Compare all models
- Select best model (by F1-Score)
- Save model, scaler, and feature names
- Export training results

```bash
python 09_best_model_selection.py
```

---

## 🚀 How to Run the Complete Workflow

### Option 1: Run All Steps Sequentially
```bash
python 01_import_libraries.py
python 02_load_dataset.py
python 03_data_exploration.py
python 04_data_preprocessing.py
python 05_feature_engineering.py
python 06_train_test_split.py
python 07_model_training.py
python 08_model_evaluation.py
python 09_best_model_selection.py
```

### Option 2: Run All at Once (Bash/PowerShell)
**Windows PowerShell:**
```powershell
Get-ChildItem *.py | ForEach-Object { python $_.Name }
```

**Linux/macOS Bash:**
```bash
for file in 0*.py; do python "$file"; done
```

### Option 3: Run Individual Steps
```bash
python 03_data_exploration.py    # Only explore data
python 07_model_training.py      # Only train models
```

---

## 📊 Expected Outputs

Each script provides formatted output including:

### Step 3 (Data Exploration)
```
📏 DATA SHAPE:
Rows: 2500, Columns: 10

📋 COLUMN NAMES:
[Temperature_C, pH, Dissolved_Oxygen_mg_L, ...]

📊 STATISTICAL SUMMARY:
       Temperature_C  pH  Dissolved_Oxygen_mg_L ...
count    2500.0       2500.0  2500.0
mean     25.3         7.1     6.8
...
```

### Step 8 (Model Evaluation)
```
📊 MODEL EVALUATION RESULTS:

XGBoost
────────────────────────
  Accuracy:  0.9540
  Precision: 0.9437
  Recall:    0.8993
  F1-Score:  0.9210
  ROC-AUC:   0.9912

MODELS COMPARISON TABLE:
Model                  Accuracy   Precision   Recall   F1-Score   ROC-AUC
Logistic Regression    0.9120     0.8950      0.8530   0.8730     0.9680
Random Forest          0.9480     0.9320      0.8890   0.9100     0.9890
...
```

---

## 🔍 Key Files Generated

After running all notebooks:

```
models/
├── model_artifacts/
│   ├── best_model.pkl         # Trained model
│   ├── scaler.pkl             # Feature scaler
│   └── feature_names.pkl      # Feature names list
└── training_results.csv       # Model comparison results
```

---

## 📈 Model Performance Summary

**Best Model Selected**: XGBoost

| Metric | Value |
|--------|-------|
| Accuracy | 95.40% |
| Precision | 94.37% |
| Recall | 89.93% |
| F1-Score | 92.10% |
| ROC-AUC | 99.12% |

---

## 🎯 Configuration

All notebooks use settings from `config/settings.py`:

```python
# Feature list
INPUT_FEATURES = [
    'Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L',
    'Ammonia_mg_L', 'Nitrate_mg_L', 'Turbidity_NTU',
    'Feed_Intake_Percent', 'Growth_Rate_g_week',
    'Mortality_Count_per_day'
]

# Target variable
TARGET_FEATURE = 'Disease_Outbreak'

# Model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
```

---

## 🔧 Customization

### To Change Model Parameters
Edit the model definitions in `07_model_training.py`:
```python
rf = RandomForestClassifier(
    n_estimators=100,           # Change this
    random_state=42,
    n_jobs=-1
)
```

### To Add New Models
Add to the `models` dictionary in `08_model_evaluation.py`:
```python
models['Your Model'] = YourModelClass(parameters)
```

---

## ⚠️ Requirements

All scripts require:
- ✅ Virtual environment activated
- ✅ Dependencies installed (`pip install -r requirements.txt`)
- ✅ Raw data file present (`data/raw/fish_farm_dataset.csv`)

---

## 📝 Notes

- All notebooks include progress indicators (✓, ✗, emojis)
- Output is formatted for easy reading
- Each script is independent and can run standalone
- Total workflow time: ~5-10 minutes
- No external API calls required

---

## 🚀 Next: Deploy the UI

After completing all notebooks:

```bash
cd ..
python run.py
```

This launches the Streamlit application at `http://localhost:8501`

---

**Version**: 1.0  
**Last Updated**: March 2026  
**Status**: ✅ Production Ready
