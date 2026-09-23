# Student Mental Health Screening System — Phase 6

Web-based screening system for the project *"Machine Learning-Based Student
Mental Health Assessment System with Comparative Model Analysis."*

## Contents

| File | Purpose |
|---|---|
| `app.py` | Streamlit web app — the screening system itself |
| `train_model.py` | Script that trains and saves the deployed model |
| `logreg_model.joblib` | Trained Logistic Regression model (pre-trained, ready to use) |
| `scaler.joblib` | StandardScaler fitted on the training data |
| `feature_columns.json` | Exact feature order the model expects |
| `metrics.json` | Test-set performance of the deployed model |
| `requirements.txt` | Python dependencies |

## Running the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

This opens the app in your browser at `http://localhost:8501`.

## Re-training the model

The model shipped here is already trained. If you want to retrain it from
the Phase 3/4 data (e.g. after data changes), run:

```bash
python train_model.py
```

It expects `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` from
`Phase 3_4/Task 2_Zainab/` (the group's finalized, leakage-free train/test
split) and writes `logreg_model.joblib`, `scaler.joblib`,
`feature_columns.json`, and `metrics.json`.

## Why Logistic Regression

Per the Phase 5 Comparative Analysis, Logistic Regression had the best
overall performance among the four models the group built (Logistic
Regression, ANN, k-NN, Naive Bayes): Accuracy 96.03%, F1-Score 52.81%,
ROC-AUC 96.13%. See the Phase 6 documentation report for the full
justification, architecture, and limitations.
