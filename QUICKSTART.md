# 🚀 Quick Start Guide

This guide explains how to use the reorganized Fish Farm Disease Prediction project.

---

## ⚡ Quick Start (3 Steps)

### Step 1: Activate Virtual Environment
```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Application
```bash
python run.py
```

The app launches at: **http://localhost:8501**

---

## 📚 Using the ML Workflow (Optional)

To see the complete machine learning process:

```bash
cd notebooks
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

Or run all at once:
```bash
PowerShell: Get-ChildItem 0*.py | % { python $_.Name }
Bash: for f in 0*.py; do python "$f"; done
```

---

## 📖 Documentation

- **Main README**: [README.md](README.md)
- **ML Workflow**: [notebooks/README.md](notebooks/README.md)

---

## 🎮 Application Features

### Dashboard Page
- Key metrics overview
- Model performance comparison
- Confusion matrix visualization

### Predictions Page
- **Manual Input**: Adjust parameters with sliders
- **Batch Upload**: Process multiple predictions from CSV
- **Quick Test**: Test with predefined scenarios

### About Page
- System information
- Technology details
- Model performance

---

## 📊 Input Parameters

The system requires 9 parameters:

| Category | Parameter | Range | Optimal |
|----------|-----------|-------|---------|
| Water Quality | Temperature (°C) | 18-34 | 24-28 |
| | pH | 5.5-8.5 | 6.5-8.0 |
| | Dissolved Oxygen (mg/L) | 2-10 | 5-10 |
| | Ammonia (mg/L) | 0-5 | 0-0.5 |
| | Nitrate (mg/L) | 0-100 | 0-40 |
| | Turbidity (NTU) | 2-100 | 2-5 |
| Fish Behavior | Feed Intake (%) | 20-100 | 80-100 |
| | Growth Rate (g/week) | 0.2-3.0 | 1.2-3.0 |
| | Mortality (per day) | 0-50 | 0-2 |

---

## 🎯 Model Performance

**Best Model**: XGBoost

| Metric | Value |
|--------|-------|
| Accuracy | 95.40% |
| Precision | 94.37% |
| Recall | 89.93% |
| F1-Score | 92.10% |
| ROC-AUC | 99.12% |

---

## 🔧 Configuration

Edit `config/settings.py` to customize:
- Model parameters
- Test/Train ratio
- File paths
- Risk thresholds

---

## 📁 Project Structure Reference

```
fish_farm_disease_prediction/
├── run.py                 # Start here
├── README.md             # Full documentation
├── app/
│   └── streamlit_app.py  # UI Application
├── notebooks/            # ML workflow
├── config/
│   └── settings.py       # Configuration
├── data/
│   └── raw/
│       └── fish_farm_dataset.csv
└── models/
    └── model_artifacts/
        ├── best_model.pkl
        └── scaler.pkl
```

---

## ⚠️ Troubleshooting

### Port 8501 in use?
```bash
streamlit run app/streamlit_app.py --server.port 8502
```

### Module not found?
```bash
pip install -r requirements.txt
```

### Model not loading?
```bash
python notebooks/09_best_model_selection.py
```

---

## ✅ What Changed

### ✨ Added
- `notebooks/` folder with 9 structured ML scripts
- `run.py` - Application launcher
- Comprehensive README.md files
- Professional project organization

### ✅ Removed
- Duplicate app files (main.py, streamlit_app_v2.py, streamlit_main_app.py)
- Faculty model files (unnecessary)
- Unprofessional keywords

### ✔ Preserved
- Full UI functionality
- All trained models
- Complete dataset
- Original source code

---

## 🎓 Learning Resources

Each notebook includes:
- Clear step-by-step workflow
- Data exploration
- Feature engineering
- Model training & evaluation
- Performance metrics

Perfect for:
- Understanding ML pipeline
- Learning data science concepts
- College projects
- Documentation & presentation

---

## 🚀 Next Steps

1. ✅ Run the application: `python run.py`
2. 📚 Explore ML workflow: `cd notebooks` and run scripts
3. 📊 Check documentation: Read README.md files
4. 🎮 Test predictions: Use the dashboard

---

**Status**: ✅ Production Ready  
**Version**: 2.0 - Reorganized & Professional  
**Last Updated**: March 2026
