from pathlib import Path
import json
import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "return_risk_model.pkl"
THRESHOLD_PATH = BASE_DIR / "models" / "risk_thresholds.json"


model = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
    thresholds = json.load(f)

HIGH_THRESHOLD = thresholds["bucket_high_at_or_above"]
MEDIUM_THRESHOLD = HIGH_THRESHOLD - 0.15


def check_return_risk(
    product_category,
    price_inr,
    discount_pct,
    payment_method,
    customer_tenure_days,
    num_previous_orders,
    num_previous_returns,
    delivery_distance_km,
    delivery_days,
    is_weekend_order,
    rating_given
):
    data = pd.DataFrame([{
        "product_category": product_category,
        "price_inr": price_inr,
        "discount_pct": discount_pct,
        "payment_method": payment_method,
        "customer_tenure_days": customer_tenure_days,
        "num_previous_orders": num_previous_orders,
        "num_previous_returns": num_previous_returns,
        "delivery_distance_km": delivery_distance_km,
        "delivery_days": delivery_days,
        "is_weekend_order": is_weekend_order,
        "rating_given": rating_given
    }])

    probability = float(model.predict_proba(data)[0][1])

    if probability >= HIGH_THRESHOLD:
        risk = "High"
    elif probability >= MEDIUM_THRESHOLD:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "return_risk_probability": round(probability, 4),
        "risk_bucket": risk
    }