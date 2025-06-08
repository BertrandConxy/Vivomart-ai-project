import streamlit as st
import pandas as pd
import plotly.express as px
from utils.feature_engineering import add_inventory_features
from utils.data_loader import load_and_clean_data
from models.risk_model import predict_risks, predict_risks_ml

from utils.metrics import (
    get_fastest_moving_products,
    get_most_wasted_products,
    get_products_expiring_soon,
    get_branch_waste_rate,
    get_category_turnover,
)

st.set_page_config(page_title="Inventory Insights", layout="wide")
st.title("🧠 Inventory Insights Dashboard")

df = load_and_clean_data("data/vivomart_inventory_sample.csv")
df = add_inventory_features(df)

# --- Summary Stats
st.subheader("📌 Summary Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Products", df["product"].nunique())
col2.metric("Total Stock Sold", int(df["stock_sold"].sum()))
col3.metric("Avg. Waste Rate", f"{df['waste_rate'].mean():.2%}")

# --- Sidebar Filters
st.sidebar.header("📅 Filter Data")
branches = st.sidebar.multiselect("Select Branch(es):", options=df["branch"].unique(), default=df["branch"].unique())
categories = st.sidebar.multiselect("Select Category(ies):", options=df["category"].unique(), default=df["category"].unique())
date_range = st.sidebar.date_input("Select Date Range:", [df["date"].min(), df["date"].max()])

# Filter data
start_date, end_date = date_range
filtered_df = df[
    (df["date"] >= pd.to_datetime(start_date)) &
    (df["date"] <= pd.to_datetime(end_date)) &
    (df["branch"].isin(branches)) &
    (df["category"].isin(categories))
]

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


# Predict risks
filtered_df = predict_risks(filtered_df)

st.subheader("🚨 Risk Overview (Rule based)")

risk_counts = filtered_df[["overstock_risk", "expiry_risk"]].sum().reset_index()
risk_counts.columns = ["Risk Type", "Count"]

fig3 = px.bar(
    risk_counts,
    x="Risk Type",
    y="Count",
    color="Risk Type",
    title="Count of Products at Risk"
)
st.plotly_chart(fig3, use_container_width=True)

st.subheader("📋 At-Risk Products")
at_risk = filtered_df[filtered_df["risk_score"] > 0][[
    "date", "product", "branch", "category", 
    "stock_received", "stock_sold", 
    "expiry_date", "overstock_risk", "expiry_risk"
]]
st.dataframe(at_risk)

# ML Prediction

df = predict_risks_ml(df)
df["Risk Label"] = df["ml_risk_prediction"].apply(lambda x: "🔴 High" if x == 1 else "🟢 Low")

# Show in dashboard
st.subheader("📊 ML Risk Prediction Results")
st.dataframe(df[["product", "category", "branch", "stock_received", "Risk Label"]])

# Charts
st.subheader("📊 Stock Received vs. Sold")
fig1 = px.bar(
    filtered_df,
    x="date",
    y=["stock_received", "stock_sold"],
    color_discrete_map={"stock_received": "green", "stock_sold": "red"},
    barmode="group",
    title="Stock Received vs Stock Sold Over Time"
)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("📉 Wastage Trends")
fig2 = px.line(
    filtered_df,
    x="date",
    y="stock_wasted",
    color="product",
    title="Wasted Stock Over Time"
)
st.plotly_chart(fig2, use_container_width=True)

high_risk_items = df[df["ml_risk_prediction"] == 1]

if not high_risk_items.empty:
    st.warning(f"⚠️ {len(high_risk_items)} high-risk items need attention!")
    st.dataframe(high_risk_items[["product", "branch"]])

st.download_button("📥 Download High-Risk Items", high_risk_items.to_csv().encode(), file_name="high_risk.csv", mime="text/csv")

