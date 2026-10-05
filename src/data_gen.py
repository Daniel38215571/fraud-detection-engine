import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta

fake = Faker("en_NG")
random.seed(42)

CITIES = ["Lagos", "Abuja", "Kano", "Ibadan", "Port Harcourt", "Benin City", "Enugu", "Kaduna"]
MERCHANTS = ["MERCH-" + str(i).zfill(4) for i in range(1, 101)]

def generate_customers(n=500):
    customers = []
    for i in range(n):
        cid = "CUST-" + str(i+1).zfill(5)
        home = random.choice(CITIES)
        normal_amount = random.randint(500000, 5000000)
        hour_start = random.randint(6, 10)
        hour_end = random.randint(18, 22)
        normal_merchants = random.sample(MERCHANTS, random.randint(3, 8))
        customers.append({
            "customer_id": cid,
            "home_city": home,
            "normal_amount_kobo": normal_amount,
            "active_hour_start": hour_start,
            "active_hour_end": hour_end,
            "normal_merchants": normal_merchants,
        })
    return customers

def generate_transactions(customers, n=10000, fraud_rate=0.05):
    transactions = []
    n_fraud = int(n * fraud_rate)
    fraud_indices = set(random.sample(range(n), n_fraud))

    for i in range(n):
        c = random.choice(customers)
        is_fraud = 1 if i in fraud_indices else 0

        amount = int(random.gauss(c["normal_amount_kobo"], c["normal_amount_kobo"] * 0.3))
        amount = max(10000, amount)
        merchant = random.choice(c["normal_merchants"])
        location = c["home_city"]
        hour = random.randint(c["active_hour_start"], c["active_hour_end"])
        date = fake.date_time_between(start_date="-90d", end_date="now")
        timestamp = date.replace(hour=hour, minute=random.randint(0, 59), second=0)

        fraud_pattern = "none"
        if is_fraud:
            fraud_pattern = random.choice(["amount_anomaly", "geo_anomaly", "off_hours", "velocity_burst"])

            if fraud_pattern == "amount_anomaly":
                amount = c["normal_amount_kobo"] * random.randint(15, 50)
            elif fraud_pattern == "geo_anomaly":
                location = random.choice([city for city in CITIES if city != c["home_city"]])
            elif fraud_pattern == "off_hours":
                hour = random.choice([1, 2, 3, 4, 23, 0])
                timestamp = timestamp.replace(hour=hour)
            elif fraud_pattern == "velocity_burst":
                pass

        transactions.append({
            "transaction_id": "TXN-" + str(i+1).zfill(6),
            "customer_id": c["customer_id"],
            "amount_kobo": amount,
            "merchant_id": merchant,
            "location": location,
            "timestamp": timestamp,
            "is_fraud": is_fraud,
            "fraud_pattern": fraud_pattern,
        })

    df = pd.DataFrame(transactions)

    extra = []
    burst_rows = df[df["fraud_pattern"] == "velocity_burst"]
    for _, row in burst_rows.iterrows():
        base_ts = row["timestamp"]
        for k in range(3):
            extra.append({
                "transaction_id": "TXN-BURST-" + row["transaction_id"] + "-" + str(k),
                "customer_id": row["customer_id"],
                "amount_kobo": row["amount_kobo"],
                "merchant_id": row["merchant_id"],
                "location": row["location"],
                "timestamp": base_ts + timedelta(seconds=30 * (k+1)),
                "is_fraud": 1,
                "fraud_pattern": "velocity_burst",
            })

    if extra:
        df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

customers = generate_customers(500)
df = generate_transactions(customers, n=10000, fraud_rate=0.05)

BASE = "/content/fraud-detection-engine"
df.to_csv(BASE + "/data/transactions.csv", index=False)

print("Total transactions: " + str(len(df)))
print("Fraudulent: " + str(int(df["is_fraud"].sum())))
print("Legitimate: " + str(int((df["is_fraud"] == 0).sum())))
print()
print("Fraud pattern breakdown:")
print(df[df["is_fraud"] == 1]["fraud_pattern"].value_counts())
print()
print("Sample:")
print(df.head(5))
