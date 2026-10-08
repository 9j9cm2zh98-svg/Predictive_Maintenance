# ── AI-Based Predictive Maintenance System ────────────────────────────────
#
# GitHub README
# =============

# AI-Based Predictive Maintenance System

> A beginner-friendly machine learning project that predicts industrial
> equipment failures using machine sensor data from the UCI AI4I 2020 dataset.

---

## Project Overview

This project simulates a **predictive maintenance** solution for steel-plant
/ rolling-mill equipment.  A **Random Forest** classifier is trained on
real-world sensor readings to predict whether a machine is likely to fail.
A **Streamlit** dashboard lets you enter live sensor values and get an
instant failure-risk prediction.

---

## Problem Statement

Unplanned equipment breakdowns are one of the biggest causes of production
losses in steel manufacturing.  Replacing parts on a fixed schedule wastes
resources; waiting for a breakdown is dangerous and expensive.

**Goal:** use sensor data (temperature, speed, torque, tool wear) to predict
failures *before* they happen, so maintenance can be scheduled at the right
time.

---

## Project Objective

- Train a machine learning model on the UCI AI4I 2020 dataset.
- Predict the binary target **Machine failure** (0 = No failure, 1 = Failure).
- Build a simple Streamlit dashboard for real-time risk prediction.
- Store prediction history in a local SQLite database.

---

## Dataset

**UCI AI4I 2020 Predictive Maintenance Dataset**

| Property         | Value              |
|------------------|--------------------|
| Source           | UCI ML Repository  |
| Rows             | 10 000             |
| Failure rate     | ~3.4 %             |
| Target variable  | `Machine failure`  |

Download:
<https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset>

Place the downloaded file at: `data/raw/ai4i2020.csv`

---

## Features Used

| Feature                    | Description                                |
|----------------------------|--------------------------------------------|
| Type                       | Machine quality class (L / M / H)          |
| Air temperature [K]        | Ambient temperature                        |
| Process temperature [K]    | Machine operating temperature              |
| Rotational speed [rpm]     | Spindle speed                              |
| Torque [Nm]                | Torque applied to the tool                 |
| Tool wear [min]            | Accumulated wear time of the current tool  |
| temp_diff *(engineered)*   | Process temp − Air temp                    |
| power *(engineered)*       | Torque × Rotational speed                  |

**Dropped columns:** `UDI`, `Product ID`, and the individual failure sub-modes
(`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) to prevent target leakage.

---

## ML Approach

| Choice              | Value / Reason                                         |
|---------------------|--------------------------------------------------------|
| Algorithm           | `RandomForestClassifier` — robust, interpretable       |
| Class imbalance     | `class_weight="balanced"` — prevents ignoring failures |
| Key metric          | **Recall** for the failure class — missing a failure is costly |
| Serialisation       | Joblib                                                 |

---

## Project Architecture

```
Sensor CSV
    ↓
src/preprocess.py   (clean, engineer features, split)
    ↓
src/train.py        (train Random Forest, evaluate, save)
    ↓
models/random_forest_model.joblib
    ↓
src/predict.py      (load model, predict single input)
    ↓
app.py              (Streamlit dashboard + SQLite history)
```

---

## Folder Structure

```
predictive_maintenance/
│
├── data/
│   ├── raw/
│   │   └── ai4i2020.csv          ← place dataset here
│   └── processed/
│       └── cleaned_data.csv      ← auto-generated
│
├── models/
│   ├── random_forest_model.joblib
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── notebooks/
│   └── 01_exploration.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/predictive_maintenance.git
cd predictive_maintenance

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Train the Model

```bash
# Make sure data/raw/ai4i2020.csv exists first!
python -m src.train
```

This will:
1. Load and preprocess `data/raw/ai4i2020.csv`
2. Save `data/processed/cleaned_data.csv`
3. Train a Random Forest classifier
4. Print evaluation metrics (accuracy, precision, recall, F1)
5. Save `models/random_forest_model.joblib`
6. Save `models/confusion_matrix.png` and `models/feature_importance.png`

---

## How to Run the Streamlit Dashboard

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**
or
deployed link - https://predictivemaintenance-bgfg5842ukmunskfqgx9bm.streamlit.app

---

## Example Workflow

```bash
# 1. Download dataset → data/raw/ai4i2020.csv
# 2. Train model
python -m src.train

# 3. (Optional) Test prediction from command line
python -m src.predict

# 4. Launch dashboard
streamlit run app.py
```

---

## Model Evaluation Results

> Metrics obtained by running `python -m src.train` on the UCI AI4I 2020 dataset.
> Test set: 2,000 rows (20% stratified split, `random_state=42`).

| Metric        | No Failure | Failure  | Overall   |
|---------------|------------|----------|-----------|
| **Accuracy**  |            |          | **98.6%** |
| **Precision** | 0.99       | **0.79** |           |
| **Recall ★**  | 0.99       | **0.81** |           |
| **F1-Score**  | 0.99       | **0.80** |           |

Confusion matrix and feature importance plots are saved to `models/`.

★ **Recall = 0.81** for the failure class means the model correctly catches
**81% of all real equipment failures** on the test set.

`class_weight="balanced"` prevents the model from learning to always predict
"no failure" (which would score ~97% accuracy but catch 0% of failures).

---

## Limitations

- This is a **prototype** — not certified industrial software.
- The dataset is synthetic (generated by a simulation), not real sensor data.
- Risk thresholds (Low / Medium / High) are **model-based categories**, not
  real safety standards.
- The model does not account for temporal patterns (time-series behaviour).
- No authentication, multi-user support, or deployment hardening.

---

## Future Improvements

- [ ] Cross-validation and hyperparameter tuning (`GridSearchCV`)
- [ ] SMOTE oversampling for better handling of class imbalance
- [ ] Time-series models (e.g., LSTM) for sequential sensor data
- [ ] Export predictions to CSV / PDF report
- [ ] Unit tests for preprocessing and prediction functions
- [ ] SQLite analytics tab in the dashboard
- [ ] Docker container for easy deployment

---

## Disclaimer

> This project is for educational and portfolio purposes only.
> It should **not** be used for real industrial safety decisions.
