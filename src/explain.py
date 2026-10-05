import pandas as pd
import numpy as np
import joblib
import shap
import json
from sklearn.model_selection import train_test_split

BASE = "/content/fraud-detection-engine"

FEATURE_COLS = [
    "amount_ratio",
    "velocity_1h",
    "geo_velocity_kmh",
    "time_deviation",
    "merchant_novelty",
]

def explain():
    df = pd.read_csv(BASE + "/data/features.csv")
    X = df[FEATURE_COLS].fillna(0)
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = joblib.load(BASE + "/src/fraud_model.pkl")

    explainer = shap.TreeExplainer(model)
    sample = X_test.sample(n=min(200, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(sample)

    if isinstance(shap_values, list):
        shap_fraud = shap_values[1]
    else:
        shap_fraud = shap_values

    y_pred = model.predict(sample)
    y_proba = model.predict_proba(sample)[:, 1]

    flagged = sample[(y_pred == 1) & (sample.index.isin(sample.index))]
    flagged_proba = model.predict_proba(flagged)[:, 1]
    flagged = flagged.assign(fraud_score=flagged_proba)

    results = []
    for i in range(min(5, len(flagged))):
        idx = flagged.index[i]
        row = flagged.iloc[i]
        pos = list(sample.index).index(idx)

        contributions = shap_fraud[pos]
        reason_pairs = sorted(
            zip(FEATURE_COLS, contributions),
            key=lambda x: -abs(x[1]),
        )

        reason_text = []
        for name, val in reason_pairs[:3]:
            direction = "raised" if val > 0 else "lowered"
            reason_text.append(
                name + " (" + direction + " score by " + str(round(val, 3)) + ")"
            )

        results.append({
            "transaction_id": df.loc[idx, "transaction_id"] if idx in df.index else "unknown",
            "fraud_score": round(float(row["fraud_score"]), 4),
            "top_reasons": reason_text,
        })

    print("SHAP explainer ready.")
    print()
    print("Top 5 flagged transactions and why they were flagged:")
    print()
    for r in results:
        print("Transaction: " + str(r["transaction_id"]))
        print("  Fraud score: " + str(r["fraud_score"]))
        print("  Reasons:")
        for reason in r["top_reasons"]:
            print("    - " + reason)
        print()

    with open(BASE + "/data/explanations.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Explanations saved to data/explanations.json")

if __name__ == "__main__":
    explain()
