import pandas as pd
import numpy as np
from datetime import timedelta

CITY_COORDS = {
    "Lagos": (6.5244, 3.3792),
    "Abuja": (9.0765, 7.3986),
    "Kano": (12.0022, 8.5920),
    "Ibadan": (7.3775, 3.9470),
    "Port Harcourt": (4.8156, 7.0498),
    "Benin City": (6.3350, 5.6037),
    "Enugu": (6.5244, 7.5103),
    "Kaduna": (10.5222, 7.4383),
}

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

def build_features(df, customers):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)

    customer_map = {c["customer_id"]: c for c in customers}
    df["normal_amount"] = df["customer_id"].map(lambda x: customer_map[x]["normal_amount_kobo"])
    df["active_start"] = df["customer_id"].map(lambda x: customer_map[x]["active_hour_start"])
    df["active_end"] = df["customer_id"].map(lambda x: customer_map[x]["active_hour_end"])
    df["normal_merchants"] = df["customer_id"].map(lambda x: customer_map[x]["normal_merchants"])

    df["amount_ratio"] = df["amount_kobo"] / df["normal_amount"]

    df["hour"] = df["timestamp"].dt.hour
    df["time_deviation"] = np.where(
        (df["hour"] >= df["active_start"]) & (df["hour"] <= df["active_end"]),
        0,
        np.minimum(
            np.abs(df["hour"] - df["active_start"]),
            np.abs(df["hour"] - df["active_end"]),
        ),
    )

    df["merchant_novelty"] = df.apply(
        lambda r: 0 if r["merchant_id"] in r["normal_merchants"] else 1, axis=1
    )

    df["prev_ts"] = df.groupby("customer_id")["timestamp"].shift(1)
    df["prev_city"] = df.groupby("customer_id")["location"].shift(1)
    df["time_delta_min"] = (df["timestamp"] - df["prev_ts"]).dt.total_seconds() / 60.0

    df["lat"] = df["location"].map(lambda x: CITY_COORDS[x][0])
    df["lon"] = df["location"].map(lambda x: CITY_COORDS[x][1])
    df["prev_lat"] = df["prev_city"].map(lambda x: CITY_COORDS[x][0] if pd.notna(x) else np.nan)
    df["prev_lon"] = df["prev_city"].map(lambda x: CITY_COORDS[x][1] if pd.notna(x) else np.nan)

    df["distance_km"] = haversine_km(df["lat"], df["lon"], df["prev_lat"], df["prev_lon"])
    df["geo_velocity_kmh"] = np.where(
        df["time_delta_min"] > 0,
        df["distance_km"] / (df["time_delta_min"] / 60.0),
        0,
    )
    df["geo_velocity_kmh"] = df["geo_velocity_kmh"].fillna(0).clip(upper=2000)

    df["velocity_1h"] = 0
    for cid, group in df.groupby("customer_id"):
        times = group["timestamp"].values
        idxs = group.index.values
        for i, t in enumerate(times):
            window_start = t - np.timedelta64(60, "m")
            count = ((times >= window_start) & (times <= t)).sum()
            df.at[idxs[i], "velocity_1h"] = count

    df["geo_velocity_kmh"] = df["geo_velocity_kmh"].fillna(0)
    return df

BASE = "/content/fraud-detection-engine"
raw = pd.read_csv(BASE + "/data/transactions.csv")

exec(open(BASE + "/src/data_gen.py").read().split("customers = generate_customers")[0])
customers = generate_customers(500)

featured = build_features(raw, customers)

FEATURE_COLS = ["amount_ratio", "velocity_1h", "geo_velocity_kmh", "time_deviation", "merchant_novelty"]
featured.to_csv(BASE + "/data/features.csv", index=False)

print("Rows: " + str(len(featured)))
print()
print("Feature summary by class:")
print(featured.groupby("is_fraud")[FEATURE_COLS].mean().round(2))
