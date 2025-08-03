# model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# ========== Load Enhanced Dataset ==========
df = pd.read_csv("data/cleaned_dataset_enhanced.csv")
df['datetime'] = pd.to_datetime(df['datetime'])

# ========== Feature Engineering (from date) ==========
df['month'] = df['datetime'].dt.month
df['day_of_week'] = df['datetime'].dt.dayofweek

# ========== Define Features and Target ==========
# Include the new lag and rolling average features
features = [
    'PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'month', 'day_of_week',
    'PM10_lag1', 'NO2_lag1', 'SO2_lag1', 'CO_lag1', 'Ozone_lag1',
    'PM10_roll_mean_3', 'NO2_roll_mean_3', 'SO2_roll_mean_3', 'CO_roll_mean_3', 'Ozone_roll_mean_3',
    'PM10_roll_mean_7', 'NO2_roll_mean_7', 'SO2_roll_mean_7', 'CO_roll_mean_7', 'Ozone_roll_mean_7'
]
target = 'AQI'

X = df[features]
y = df[target]

# ========== Train-Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== Model Training with Hyperparameter Tuning ==========
print("--- Training Tuned Random Forest ---")

# Define the parameter grid for Randomized Search
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Initialize the model and the search
rf = RandomForestRegressor(random_state=42)
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,  # Number of parameter settings that are sampled
    cv=5,       # 5-fold cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1   # Use all available cores
)

# Fit the random search model
random_search.fit(X_train, y_train)

# Get the best model
best_rf_model = random_search.best_estimator_
print(f"\nBest Random Forest Parameters: {random_search.best_params_}")


# ========== Model Comparison: XGBoost ==========
print("\n--- Training XGBoost for Comparison ---")
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)


# ========== Model Evaluation ==========
# Evaluate the tuned Random Forest
print("\n--- Tuned Random Forest Evaluation ---")
y_pred_rf = best_rf_model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred_rf):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred_rf):.2f}")

# Evaluate XGBoost
print("\n--- XGBoost Evaluation ---")
y_pred_xgb = xgb_model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred_xgb):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_xgb)):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred_xgb):.2f}")


# ========== Save the Best Model ==========
# For this example, we save the tuned Random Forest model
joblib.dump(best_rf_model, 'models/air_quality_model_tuned.pkl')
print("\nTuned model saved to 'models/air_quality_model_tuned.pkl'")
