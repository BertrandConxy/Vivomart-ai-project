import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

from utils.data_loader import load_and_clean_data
from utils.feature_engineering import add_ml_features, add_inventory_features

df = load_and_clean_data('data/vivomart_inventory_sample.csv')
df = add_inventory_features(df)
df = add_ml_features(df)

df["Risk_Label"] = (
    (df["days_until_expiry"] <= 5) | (df["sales_ratio"] < 0.3)
).astype(int)

features = ["stock_received", "stock_sold", "stock_wasted", 
            "days_until_expiry", "sales_ratio", "waste_ratio", 
            "day_of_week", "is_weekend"]

X = df[features]
y = df["Risk_Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, "models/risk_predictor.pkl")


