"""
Train the final Logistic Regression model for the Student Mental Health
Assessment System (Phase 6), using the same corrected, leakage-free
feature set and train/test split produced in Phase 3/4 by the group.

Logistic Regression is used because it was the top performer in Phase 5
Comparative Analysis (Accuracy 96.03%, F1 52.81%, ROC-AUC 96.13%).
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)

DATA_DIR = "../../project/ML PROJECT/Phase 3_4/Task 2_Zainab"

X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").values.ravel()
y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").values.ravel()

feature_columns = list(X_train.columns)
print("Num features:", len(feature_columns))

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(max_iter=2000, class_weight=None, random_state=42)
model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)
y_proba = model.predict_proba(X_test_s)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba),
}
cm = confusion_matrix(y_test, y_pred).tolist()

print(json.dumps(metrics, indent=2))
print("Confusion matrix:", cm)

joblib.dump(model, "logreg_model.joblib")
joblib.dump(scaler, "scaler.joblib")
with open("feature_columns.json", "w") as f:
    json.dump(feature_columns, f, indent=2)
with open("metrics.json", "w") as f:
    json.dump({"metrics": metrics, "confusion_matrix": cm}, f, indent=2)

print("Saved model, scaler, feature_columns.json, metrics.json")
