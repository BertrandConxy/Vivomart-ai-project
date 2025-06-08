import pandas as pd

def add_inventory_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Rename columns to lowercase with underscores for consistency
    df.rename(columns={
        'Date': 'date',
        'Branch': 'branch',
        'Product': 'product',
        'Category': 'category',
        'Opening_Stock': 'opening_stock',
        'Stock_Received': 'stock_received',
        'Stock_Sold': 'stock_sold',
        'Stock_Wasted': 'stock_wasted',
        'Expiry_Date': 'expiry_date'
    }, inplace=True)

    # Add engineered features
    df['net_stock_change'] = df['stock_received'] - df['stock_sold'] - df['stock_wasted']

    total_stock = df['opening_stock'] + df['stock_received']
    df['stock_turnover_rate'] = (df['stock_sold'] / total_stock).round(2)
    df['waste_rate'] = (df['stock_wasted'] / total_stock).round(3)

    df['days_to_expiry'] = (df['expiry_date'] - df['date']).dt.days
    df['is_expiring_soon'] = df['days_to_expiry'] <= 3

    return df

def add_ml_features(df):
    df = df.copy()
    df["days_until_expiry"] = (df["expiry_date"] - df["date"]).dt.days
    df["sales_ratio"] = df["stock_sold"] / (df["stock_received"] + 1)
    df["waste_ratio"] = df["stock_wasted"] / (df["stock_received"] + 1)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df
