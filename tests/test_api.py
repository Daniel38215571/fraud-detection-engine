import sys
import os

BASE = "/content/fraud-detection-engine"
sys.path.insert(0, BASE)
sys.path.insert(0, BASE + "/src")

import pytest
from fastapi.testclient import TestClient
from api.main import app, classify

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_clear_fraud_scores_high():
    response = client.post("/score", json={
        "transaction_id": "TXN-FRAUD-001",
        "amount_ratio": 42.0,
        "velocity_1h": 12,
        "geo_velocity_kmh": 850,
        "time_deviation": 8,
        "merchant_novelty": 1,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["fraud_score"] > 0.85
    assert data["classification"] == "LIKELY_FRAUD"
    assert len(data["reasons"]) >= 3


def test_clear_legitimate_scores_low():
    response = client.post("/score", json={
        "transaction_id": "TXN-LEGIT-001",
        "amount_ratio": 1.0,
        "velocity_1h": 1,
        "geo_velocity_kmh": 0,
        "time_deviation": 0,
        "merchant_novelty": 0,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["fraud_score"] < 0.5
    assert data["classification"] == "LEGITIMATE"


def test_negative_amount_ratio_rejected():
    response = client.post("/score", json={
        "transaction_id": "TXN-BAD-001",
        "amount_ratio": -5.0,
        "velocity_1h": 1,
        "geo_velocity_kmh": 5,
        "time_deviation": 0,
        "merchant_novelty": 0,
    })
    assert response.status_code == 422


def test_invalid_merchant_novelty_rejected():
    response = client.post("/score", json={
        "transaction_id": "TXN-BAD-002",
        "amount_ratio": 1.0,
        "velocity_1h": 1,
        "geo_velocity_kmh": 5,
        "time_deviation": 0,
        "merchant_novelty": 5,
    })
    assert response.status_code == 422


def test_missing_field_rejected():
    response = client.post("/score", json={
        "transaction_id": "TXN-BAD-003",
        "amount_ratio": 1.0,
    })
    assert response.status_code == 422


def test_classify_thresholds():
    assert classify(0.1) == "LEGITIMATE"
    assert classify(0.49) == "LEGITIMATE"
    assert classify(0.5) == "SUSPICIOUS"
    assert classify(0.84) == "SUSPICIOUS"
    assert classify(0.85) == "LIKELY_FRAUD"
    assert classify(0.99) == "LIKELY_FRAUD"


def test_reasons_generated_for_anomalies():
    response = client.post("/score", json={
        "transaction_id": "TXN-REASON-001",
        "amount_ratio": 30.0,
        "velocity_1h": 1,
        "geo_velocity_kmh": 5,
        "time_deviation": 0,
        "merchant_novelty": 0,
    })
    assert response.status_code == 200
    reasons = response.json()["reasons"]
    assert any("Amount" in r for r in reasons)
