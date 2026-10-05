import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

BASE = "/content/fraud-detection-engine"

FEATURE_COLS = [
    "amount_ratio",
    "velocity_1h",
    "geo_velocity_kmh",
    "time_deviation",
    "merchant_novelty",
]

def train():
    df = pd.read_csv(BASE + "/data/features.csv")

    X = df[FEATURE_COLS].fillna(0)
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "auc_roc": round(roc_auc_score(y_test, y_proba), 4),
        "test_size": int(len(y_test)),
        "fraud_in_test": int(y_test.sum()),
    }

    print("Model retrained without class weights.")
    print()
    print("Precision: " + str(metrics["precision"]))
    print("Recall:    " + str(metrics["recall"]))
    print("F1:        " + str(metrics["f1"]))
    print("AUC-ROC:   " + str(metrics["auc_roc"]))
    print()
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, BASE + "/src/fraud_model.pkl")
    with open(BASE + "/data/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print()
    print("Model saved.")
    print()
    print("Probability samples:")
    print("  Normal transaction (all signals low): " + str(round(float(model.predict_proba([[1.1, 1, 5, 0, 0]])[0][1]), 4)))
    print("  Obvious fraud (all signals high):     " + str(round(float(model.predict_proba([[42.0, 12, 850, 8, 1]])[0][1]), 4)))

if __name__ == "__main__":
    train()
