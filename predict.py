# predict.py
import joblib
import pandas as pd
from datetime import datetime

# --- Load Model and Full Historical Data ---
try:
    model = joblib.load('models/air_quality_model_tuned.pkl')
    df_historical = pd.read_csv("data/cleaned_dataset_enhanced.csv")
    df_historical['datetime'] = pd.to_datetime(df_historical['datetime'])
except FileNotFoundError as e:
    print(f"Error loading a required file: {e}")
    print("Please ensure 'air_quality_model_tuned.pkl' is in the 'models' folder and")
    print("'cleaned_dataset_enhanced.csv' is in the 'data' folder before running.")
    exit()


# --- Feature Preparation Function (remains the same) ---
base_features = ['PM10', 'NO2', 'SO2', 'CO', 'Ozone']
model_features = model.feature_names_in_ # Get feature names from the loaded model

def prepare_features_for_prediction(current_day_data, historical_data):

    current_df = pd.DataFrame([current_day_data])
    combined_df = pd.concat([historical_data, current_df], ignore_index=True)
    combined_df['datetime'] = pd.to_datetime(combined_df['datetime'])
    
    for pollutant in base_features:
        combined_df[f'{pollutant}_lag1'] = combined_df[pollutant].shift(1)
        combined_df[f'{pollutant}_roll_mean_3'] = combined_df[pollutant].rolling(window=3, min_periods=1).mean()
        combined_df[f'{pollutant}_roll_mean_7'] = combined_df[pollutant].rolling(window=7, min_periods=1).mean()
        
    combined_df['month'] = combined_df['datetime'].dt.month
    combined_df['day_of_week'] = combined_df['datetime'].dt.dayofweek
    
    prediction_features = combined_df.iloc[-1:]
    
    return prediction_features[model_features]

# --- NEW: Function to estimate pollutant levels ---
def estimate_pollutants_for_date(prediction_date, historical_df):

    month = prediction_date.month
    day = prediction_date.day
    
    # Find data for the same month and day in previous years
    similar_days_data = historical_df[
        (historical_df['datetime'].dt.month == month) & 
        (historical_df['datetime'].dt.day == day)
    ]
    
    if similar_days_data.empty:
        print(f"Warning: No historical data found for {month}-{day}. Using overall monthly average.")
        similar_days_data = historical_df[historical_df['datetime'].dt.month == month]
        if similar_days_data.empty:
            # Fallback to the last available record if no monthly data exists (unlikely)
            return historical_df.iloc[-1][base_features].to_dict()

    # Calculate the mean of the base pollutants
    estimated_values = similar_days_data[base_features].mean().to_dict()
    return estimated_values


# --- Main Interactive Prediction Logic ---
def run_interactive_prediction():
    print("--- AQI Forecasting Tool ---")
    
    # 1. Get date input from user
    while True:
        date_str = input("Enter the future date for prediction (YYYY-MM-DD): ")
        try:
            prediction_date = pd.to_datetime(date_str)
            break
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")

    # 2. Estimate pollutant values for the selected date
    print("\nEstimating pollutant values based on historical averages...")
    estimated_inputs = estimate_pollutants_for_date(prediction_date, df_historical)
    today_input = {'datetime': prediction_date, **estimated_inputs}
    
    print("Estimated Pollutant Values:")
    for key, value in estimated_inputs.items():
        print(f"  - {key}: {value:.2f}")

    # 3. Find the 7 days of historical data before the user's chosen date
    historical_subset = df_historical[df_historical['datetime'] < prediction_date].tail(7)
    
    if len(historical_subset) < 7:
        print("\nWarning: Could not find 7 full days of historical data before the selected date.")
        print("Prediction might be less accurate.")
    
    # 4. Prepare the full feature set
    input_df = prepare_features_for_prediction(today_input, historical_subset)
    
    print("\nGenerated features for prediction:")
    print(input_df.to_string())

    # 5. Predict
    prediction = model.predict(input_df)

    print("\n-----------------------------------------")
    print(f"Predicted AQI for {date_str}: {prediction[0]:.2f}")
    print("-----------------------------------------")


if __name__ == "__main__":
    run_interactive_prediction()
