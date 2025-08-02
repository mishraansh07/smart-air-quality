import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

# ========== Load Dataset ==========
df = pd.read_csv("data/cleaned_dataset.csv")

# Check actual column names (debug)
print(df.columns)

# Fix: Use correct column name
df['datetime'] = pd.to_datetime(df['datetime'])  # Use lowercase if that's the actual name

# ========== Feature Engineering ==========
df['month'] = df['datetime'].dt.month
df['day_of_week'] = df['datetime'].dt.dayofweek  # Changed column name to match feature list

# ========== Define Features and Target ==========
features = ['PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'month', 'day_of_week']
target = 'PM2.5'

X = df[features]
y = df[target]

# ========== Train-Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== Model Training ==========
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ========== Model Evaluation ==========
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# ========== Save Model ==========
joblib.dump(model, 'models/air_quality_model.pkl')
print("Model saved to 'models/air_quality_model.pkl'")