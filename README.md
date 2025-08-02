# 🌫️ Smart Air Quality

An AI-powered PM2.5 prediction engine using Random Forest Regression. This project uses cleaned air quality datasets and machine learning to predict pollutant concentration (PM2.5) based on temporal and weather-related features.

---

## 📊 Features

- Predicts **PM2.5** concentration using a trained Random Forest model
- Preprocessed and cleaned dataset
- Includes model evaluation: MAE, RMSE, R²
- CLI-based prediction engine
- Ready for deployment or integration with dashboards

---

## 🧠 Model Performance

- **Mean Absolute Error (MAE)**: 16.78 µg/m³  
- **Root Mean Squared Error (RMSE)**: 27.55 µg/m³  
- **R² Score**: 0.88  
- ⚠️ Note: These metrics may vary depending on dataset version

---

## 📁 Project Structure
smart-air-quality/
│
├── cleaned_dataset.csv # Preprocessed dataset
├── model.pkl # Trained Random Forest model (saved using joblib)
├── train.py # Script to train and evaluate model
├── predict.py # CLI prediction engine
├── requirements.txt # Python dependencies
└── README.md # You're reading it :)


---

## 🔧 Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/mishraansh07/smart-air-quality.git
cd smart-air-quality

pip install -r requirements.txt

python train.py

input_data = {
    'day_of_week': 2,
    'month': 8,
    'hour': 14,
    'humidity': 60,
    'temperature': 31
}

python predict.py

🔮 Predicted PM2.5: 69.33 µg/m³

