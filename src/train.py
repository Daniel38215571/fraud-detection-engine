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
        class_weight="balanced",
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

    print("Model trained.")
    print()
    print("Test set size: " + str(metrics["test_size"]))
    print("Fraud cases in test set: " + str(metrics["fraud_in_test"]))
    print()
    print("Precision: " + str(metrics["precision"]))
    print("Recall:    " + str(metrics["recall"]))
    print("F1:        " + str(metrics["f1"]))
    print("AUC-ROC:   " + str(metrics["auc_roc"]))
    print()
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()
    print("Feature importances:")
    for name, importance in sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda x: -x[1],
    ):
        print("  " + name.ljust(20) + ": " + str(round(importance, 4)))

    joblib.dump(model, BASE + "/src/fraud_model.pkl")
    with open(BASE + "/data/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print()
    print("Model saved to src/fraud_model.pkl")
    print("Metrics saved to data/metrics.json")

    return model, metrics

if __name__ == "__main__":
    train()
