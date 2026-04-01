# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree (Udacity)

---

## Project Description

This project builds a machine learning pipeline to identify credit card customers most likely to churn. It refactors an exploratory Jupyter notebook into a production-quality Python package with proper testing, logging, and code quality standards.

The pipeline trains two classifiers — a Logistic Regression and a Random Forest — on bank customer data, evaluates their performance, and saves the trained models for downstream use.

---

## Files and Data Description

### Main Files

- `churn_library.py`  
  Core ML pipeline library. Contains all functions for data loading, EDA, feature engineering, model training, and evaluation.

- `churn_script_logging_and_tests.py`  
  Unit tests for every function in `churn_library.py`. Logs INFO and ERROR messages to `logs/churn_library.log`.

- `churn_notebook.ipynb`  
  Original reference notebook containing the working solution before refactoring.

---

### Data

- `data/bank_data.csv`  
  10,127 bank customer records with 21 features including demographics, account behaviour, and transaction history. The target column is `Attrition_Flag` (converted to binary `Churn`: 0 = Existing Customer, 1 = Attrited Customer).

---

### Output Directories

After running the project, outputs will be saved to:

- EDA images → `images/eda/`
- Model results → `images/results/`
- Models → `models/`
- Logs → `logs/churn_library.log`

---

## Running the Files

### Install dependencies

```bash
pip install -r requirements.txt
```

### 1. Run the Pipeline

```bash
python churn_library.py
```

Loads the data, performs EDA, encodes features, trains both models, saves evaluation plots, and stores trained models as `.pkl` files.

---

### 2. Run Tests and Logging

```bash
python churn_script_logging_and_tests.py
```

Runs all unit tests and writes detailed INFO/ERROR messages to `logs/churn_library.log`. Tests validate that output files (EDA images, result plots, model files) are created correctly.

---

### 3. Code Quality

```bash
# Auto-format
autopep8 --in-place --aggressive --aggressive churn_library.py
autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py

# Check lint score (target: > 7.0)
pylint churn_library.py
pylint churn_script_logging_and_tests.py
```

---

## Expected Outputs

- Models:
  - `models/rfc_model.pkl`
  - `models/logistic_model.pkl`

- EDA plots in `images/eda/`:
  - `churn_distribution.png`
  - `customer_age_distribution.png`
  - `marital_status_distribution.png`
  - `total_trans_ct_distribution.png`
  - `heatmap.png`

- Model evaluation plots in `images/results/`:
  - `rf_results.png`
  - `logistic_results.png`
  - `feature_importances.png`
  - `roc_curve_result.png`

- Logs:
  - `logs/churn_library.log`
