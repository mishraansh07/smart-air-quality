# main.py
import pandas as pd
import seaborn as sns
import os

# ========== Load Dataset ==========
file_path = "data/final_dataset.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found at {file_path}")

df = pd.read_csv(file_path)

print(f"Original Dataset Shape: {df.shape}")
print("Missing Values per Column:\n", df.isnull().sum())

# ========== Datetime Handling ==========
df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Date']].rename(columns={'Date': 'day'}))
# Sort by date, which is crucial for time-series features
df = df.sort_values('datetime').reset_index(drop=True)

# ========== Pollutant Columns ==========
pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone']
existing_pollutants = [col for col in pollutants if col in df.columns]

# Forward fill missing values in existing pollutant columns
df[existing_pollutants] = df[existing_pollutants].ffill()

# ========== Clean Up Columns ==========
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]


# ========== NEW: Advanced Feature Engineering ==========

# 1. Lag Features
# Create features from the previous day's data
for pollutant in ['PM10', 'NO2', 'SO2', 'CO', 'Ozone']:
    df[f'{pollutant}_lag1'] = df[pollutant].shift(1)

# 2. Rolling Averages
# Create features based on the average of the last 3 and 7 days
for pollutant in ['PM10', 'NO2', 'SO2', 'CO', 'Ozone']:
    df[f'{pollutant}_roll_mean_3'] = df[pollutant].rolling(window=3).mean()
    df[f'{pollutant}_roll_mean_7'] = df[pollutant].rolling(window=7).mean()

# Drop rows with NaN values created by lag/rolling features
df.dropna(inplace=True)
print(f"Shape after adding features and dropping NaNs: {df.shape}")

# ========== Saving New Enhanced Dataset ==========
os.makedirs('data', exist_ok=True)
df.to_csv("data/cleaned_dataset_enhanced.csv", index=False)
print("Saved enhanced dataset to data/cleaned_dataset_enhanced.csv")
