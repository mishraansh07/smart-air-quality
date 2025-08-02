import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load('models/air_quality_model.pkl')

# Use the correct feature names
feature_names = ['PM10', 'NO2', 'SO2', 'CO', 'Ozone', 'month', 'day_of_week']

# Sample input using correct keys
sample_input = {
    'PM10': 77.2,
    'NO2': 43.3,
    'SO2': 462,
    'CO': 1.2,
    'Ozone': 30.7,
    'month': 12,         # lowercase
    'day_of_week': 2     # lowercase
}

# Convert to DataFrame
input_df = pd.DataFrame([sample_input], columns=feature_names)

# Predict
prediction = model.predict(input_df)

print(f"🔮 Predicted PM2.5: {prediction[0]:.2f} µg/m³")
