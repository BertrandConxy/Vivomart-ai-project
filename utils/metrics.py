def get_fastest_moving_products(df, top_n=5):
    result = (
        df.groupby("product")["stock_sold"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    result.columns = ["product", "total_sold"]
    return result

def get_most_wasted_products(df, top_n=5):
    result = (
        df.groupby("product")["stock_wasted"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    result.columns = ["product", "total_wasted"]
    return result

def get_products_expiring_soon(df, within_days=3):
    return df[df["days_to_expiry"] <= within_days][
        ["date", "product", "branch", "days_to_expiry", "expiry_date"]
    ].sort_values(by="days_to_expiry")

def get_branch_waste_rate(df):
    return (
        df.groupby("branch")["waste_rate"]
        .mean()
        .round(3)
        .reset_index()
        .rename(columns={"waste_rate": "avg_waste_rate"})
    )

def get_category_turnover(df):
    return (
        df.groupby("category")["stock_turnover_rate"]
        .mean()
        .round(2)
        .reset_index()
        .rename(columns={"stock_turnover_rate": "avg_turnover_rate"})
    )
