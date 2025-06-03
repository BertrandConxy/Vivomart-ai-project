import pandas as pd

def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Convert date fields
    df['date'] = pd.to_datetime(df['date'], format="%d/%m/%Y")
    df['expiry_date'] = pd.to_datetime(df['expiry_date'], format="%d/%m/%Y")

    # Sort data
    df = df.sort_values(by=['branch', 'product', 'date'])

    # Create stock_remaining
    df['stock_remaining'] = (
        df['opening_stock'] + df['stock_received'] - df['stock_sold'] - df['stock_wasted']
    )

    return df