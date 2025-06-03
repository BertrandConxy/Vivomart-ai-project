from utils.data_loader import load_and_clean_data
from utils.feature_engineering import add_inventory_features

df = load_and_clean_data('data/vivomart_inventory_sample.csv')

df = add_inventory_features(df)
print(df.head())

# print(df.head())
# print(df.columns)
# print(df.describe())