import os
import sys
import joblib
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

BASE = "/content/fraud-detection-engine"
sys.path.insert(0, BASE + "/src")

FEATURE_COLS = [
    "amount_ratio",
    "velocity_1h",
    "geo_velocity_kmh",
    "time_deviation",
    "merchant_novelty",
]

app = FastAPI(
    title="Fraud Detection API",
    description="Scores transactions for fraud risk with explainable signals.",
    version="0.1.0",
)

MODEL_PATH = BASE + "/src/fraud_model.pkl"
if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model not found. Run train.py first.")

model = joblib.load(MODEL_PATH)


class Transaction(BaseModel):
    transaction_id: str
    amount_ratio: float = Field(..., ge=0)
    velocity_1h: float = Field(..., ge=0)
    geo_velocity_kmh: float = Field(..., ge=0)
    time_deviation: float = Field(..., ge=0)
    merchant_novelty: int = Field(..., ge=0, le=1)


class FraudScore(BaseModel):
    transaction_id: str
    fraud_score: float
    classification: str
    reasons: List[str]
    scored_at: str


def classify(score: float) -> str:
    if score < 0.5:
        return "LEGITIMATE"
    elif score < 0.85:
        return "SUSPICIOUS"
    else:
        return "LIKELY_FRAUD"


def build_reasons(row: dict) -> List[str]:
    reasons = []
    if row["amount_ratio"] > 5:
        reasons.append("Amount is " + str(round(row["amount_ratio"], 1)) + "x customer normal")
    if row["geo_velocity_kmh"] > 900:
        reasons.append("Geo-velocity " + str(round(row["geo_velocity_kmh"])) + " km/h is physically implausible")
    if row["velocity_1h"] > 5:
        reasons.append("Velocity " + str(int(row["velocity_1h"])) + " transactions in the last hour")
    if row["time_deviation"] > 4:
        reasons.append("Transaction at " + str(int(row["time_deviation"])) + " hours outside active window")
    if row["merchant_novelty"] == 1:
        reasons.append("New merchant for this customer")
    if not reasons:
        reasons.append("No anomaly signals fired")
    return reasons


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/score", response_model=FraudScore)
def score_transaction(txn: Transaction):
    try:
        features = pd.DataFrame([{
            "amount_ratio": txn.amount_ratio,
            "velocity_1h": txn.velocity_1h,
            "geo_velocity_kmh": txn.geo_velocity_kmh,
            "time_deviation": txn.time_deviation,
            "merchant_novelty": txn.merchant_novelty,
        }])
        proba = float(model.predict_proba(features)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail="Scoring failed: " + str(e))

    return FraudScore(
        transaction_id=txn.transaction_id,
        fraud_score=round(proba, 4),
        classification=classify(proba),
        reasons=build_reasons(txn.model_dump()),
        scored_at=datetime.utcnow().isoformat(),
    )
