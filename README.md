# Fraud Detection Engine

Explainable fraud detection for financial transactions. Five engineered signals feeding a Random Forest classifier. Every score comes with the signals that drove it.

## Design Principle

**Explainability is not optional in a regulated industry.**

A fraud analyst cannot act on a black-box score. If a transaction is flagged, the customer calls. The analyst must defend the decision to the customer and to a regulator. This engine outputs the score and the reasons.

## The Five Signals

1. **Amount anomaly** — current amount vs. customer's median
2. **Velocity** — transactions in the last hour
3. **Geo-velocity** — distance over time between consecutive transactions
4. **Time-of-day deviation** — hours outside the customer's active window
5. **Merchant novelty** — whether the merchant is new for this customer

## Classification Zones

| Score Range | Classification | Action |
|---|---|---|
| 0.0 – 0.3 | LEGITIMATE | No action |
| 0.3 – 0.7 | SUSPICIOUS | Manual review |
| 0.7 – 1.0 | LIKELY_FRAUD | Block |

## Structure

    src/
        data_gen.py     Synthetic transaction generator
        features.py     Feature engineering (five signals)
        train.py        Model training and evaluation
        explain.py      SHAP-based per-prediction explanations
    api/
        main.py         FastAPI scoring service
    tests/
        test_api.py     Pytest suite (8 tests)

## Stack

- Python 3.10+
- scikit-learn (Random Forest)
- SHAP (explainability)
- FastAPI
- Pandas, NumPy
- Pytest

## Metrics

The model is evaluated on precision, recall, F1, and AUC-ROC. Accuracy is not reported because the 5% fraud base rate makes accuracy meaningless.

## Endpoints

    GET  /health          Liveness check
    POST /score           Score a transaction

Request body:

    {
      "transaction_id": "TXN-001",
      "amount_ratio": 42.0,
      "velocity_1h": 12,
      "geo_velocity_kmh": 850,
      "time_deviation": 8,
      "merchant_novelty": 1
    }

Response:

    {
      "transaction_id": "TXN-001",
      "fraud_score": 0.98,
      "classification": "LIKELY_FRAUD",
      "reasons": [
        "Amount is 42.0x customer's normal",
        "Geo-velocity 850 km/h is physically implausible",
        "Velocity 12 transactions in the last hour"
      ],
      "scored_at": "2026-10-05T..."
    }

## Author

Anuoluwapo Daniel Ojo
Software Engineer | Fintech & ML

- GitHub: github.com/Daniel38215571
- LinkedIn: linkedin.com/in/daniel-ojo-879273197
- Email: ojodaniel38@gmail.com
- Location: Lagos, Nigeria (Remote-ready)

## Known Limitations

During diagnostic testing, two limitations were identified:

- **Velocity Burst First-Transaction Blind Spot:** The model achieves 100% recall on subsequent transactions within a velocity burst but consistently misses the first transaction of the burst (mean score: 0.23). This occurs because the first transaction appears normal in isolation and does not yet contain the temporal context created by the subsequent rapid transactions.
- **Inactive Merchant Novelty Feature:** The `merchant_novelty` feature is currently producing all-zero values (mean: 0.0, std: 0.0), indicating that the feature is not contributing meaningful signal to the model. This points to an issue in the feature-engineering pipeline rather than necessarily a limitation of the model itself.

## Future Enhancements

- Fix and validate the `merchant_novelty` feature in `src/features.py`.
- Engineer a "burst-start" feature to identify the first transaction in a rapid sequence.
- Evaluate sequence-based approaches, such as LSTM models, for capturing temporal anomaly patterns.
- Expand testing to cover additional temporal and edge-case scenarios.
