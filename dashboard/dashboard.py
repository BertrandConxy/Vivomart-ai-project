import streamlit as st
import pandas as pd
from utils.feature_engineering import add_inventory_features
from utils.metrics import (
    get_fastest_moving_products,
    get_most_wasted_products,
    get_products_expiring_soon,
    get_branch_waste_rate,
    get_category_turnover,
)

st.set_page_config(page_title="Inventory Insights", layout="wide")
st.title("🧠 Inventory Insights Dashboard")

# Load your main CSV
df = pd.read_csv("data/vivomart_inventory_sample.csv", parse_dates=["date", "expiry_date"])
df = add_inventory_features(df)

# --- Summary Stats
st.subheader("📌 Summary Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Products", df["product"].nunique())
col2.metric("Total Stock Sold", int(df["stock_sold"].sum()))
col3.metric("Avg. Waste Rate", f"{df['waste_rate'].mean():.2%}")

# --- Fast Moving
st.subheader("🚀 Top 5 Fastest Moving Products")
st.dataframe(get_fastest_moving_products(df), use_container_width=True)

# --- Wasted
st.subheader("♻️ Top 5 Most Wasted Products")
st.dataframe(get_most_wasted_products(df), use_container_width=True)

# --- Expiring Soon
st.subheader("⏰ Products Expiring Soon")
expiring_df = get_products_expiring_soon(df)
if expiring_df.empty:
    st.success("✅ No products expiring in the next few days!")
else:
    st.warning("⚠️ Some products are expiring soon!")
    st.dataframe(expiring_df, use_container_width=True)

# --- Waste Rate by Branch
st.subheader("🏪 Branch-wise Waste Rate")
st.dataframe(get_branch_waste_rate(df), use_container_width=True)

# --- Turnover by Category
st.subheader("📦 Category-wise Stock Turnover")
st.dataframe(get_category_turnover(df), use_container_width=True)
