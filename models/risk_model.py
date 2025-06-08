import pandas as pd
from datetime import timedelta

import joblib

from utils.feature_engineering import add_ml_features

model = joblib.load("models/risk_predictor.pkl")

def predict_risks_ml(df):
    df = df.copy()
    df = add_ml_features(df)

    features = ["stock_received", "stock_sold", "stock_wasted", 
                "days_until_expiry", "sales_ratio", "waste_ratio", 
                "day_of_week", "is_weekend"]

    df["ml_risk_prediction"] = model.predict(df[features])
    return df

def predict_risks(df: pd.DataFrame, overstock_ratio=0.3, expiry_days_threshold=5) -> pd.DataFrame:
    """
    Add risk flags to the dataframe.
    
    - Overstock Risk: High stock received but low sales.
    - Expiry Risk: Product expiry date is approaching.
    """

    df = df.copy()
    today = pd.Timestamp.today()

    # Add overstock risk (low sales compared to received)
    df["overstock_risk"] = ((df["stock_sold"] / (df["stock_received"] + 1)) < overstock_ratio)

    # Add expiry risk (expiring in next `expiry_days_threshold` days)
    df["expiry_risk"] = (df["expiry_date"] - today).dt.days <= expiry_days_threshold

    # Final Risk Score (simple sum for now)
    df["risk_score"] = df[["overstock_risk", "expiry_risk"]].sum(axis=1)

    return df
