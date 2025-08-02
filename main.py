import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ========== Load Dataset ==========
file_path = "data/final_dataset.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found at {file_path}")

df = pd.read_csv(file_path)

print(f"Dataset Shape: {df.shape}")
print("Missing Values per Column:\n", df.isnull().sum())

# ========== Datetime Handling ==========
# Rename 'Date' column to 'day' for compatibility
df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Date']].rename(columns={'Date': 'day'}))

# ========== Pollutant Columns ==========
pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone']
existing_pollutants = [col for col in pollutants if col in df.columns]

# Forward fill missing values in existing pollutant columns
df[existing_pollutants] = df[existing_pollutants].ffill()

# ========== Clean Up Columns ==========
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# ========== Visualization ==========

# 1. PM2.5 Trend Plot
plt.figure(figsize=(14, 5))
plt.plot(df['datetime'], df['PM2.5'], color='tab:red')
plt.title('PM2.5 Levels Over Time')
plt.xlabel('Date')
plt.ylabel('PM2.5')
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Correlation Heatmap of Pollutants
plt.figure(figsize=(8, 6))
sns.heatmap(df[existing_pollutants].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Between Pollutants')
plt.tight_layout()
plt.show()


# Saving New Dataset
# Save cleaned dataset (optional)
df.to_csv("data/cleaned_dataset.csv")
# ========== Summary Statistics ==========
summary_stats = df.describe()