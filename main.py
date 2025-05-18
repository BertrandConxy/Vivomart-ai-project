import pandas as pd

# Load dataset
df = pd.read_csv('data/vivomart_inventory_sample.csv')

# Preview
print(df.head())
print(df.columns)
print(df.describe())
