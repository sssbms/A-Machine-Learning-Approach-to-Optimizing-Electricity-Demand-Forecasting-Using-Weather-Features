# === IMPORT LIBRARIES ===
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# === LOAD & PREPARE DATA ===
@st.cache_data
def load_data():
    df = pd.read_csv("C:\\Users\\User11\\Downloads\\ML ASSIGN\\KD34403 ML PROJECT\\KD34403 ML PROJECT\\final_merged_objective2.csv")
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['Renewables'] = df['Solar'] + df['Wind']
    return df

df = load_data()

# === FEATURES & TARGET ===
features = ['Solar', 'Wind', 'Renewables', 'Imports', 'temp', 'humidity', 'hour', 'month']
target = 'Load'
X = df[features].copy()
y = df[target]
X = X.fillna(X.mean())

# === TRAIN-TEST SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === RANDOM FOREST TUNING ===
@st.cache_resource
def train_model():
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
    }
    rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=3, scoring='r2', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    return rf_grid.best_estimator_

rf_best = train_model()

# === STREAMLIT APP ===
st.title("⚡️ Electricity Load Prediction App")
st.markdown("This app uses a Random Forest model to predict electricity demand based on environmental and generation inputs.")

st.header("🔧 Input Parameters")

# Use sliders for user input
solar = st.slider("Solar Generation (MW)", 0.0, 5000.0, 1000.0)
wind = st.slider("Wind Generation (MW)", 0.0, 5000.0, 1000.0)
imports = st.slider("Imports (MW)", 0.0, 10000.0, 2000.0)
temp = st.slider("Temperature (°C)", -10.0, 45.0, 25.0)
humidity = st.slider("Humidity (%)", 0.0, 100.0, 50.0)
hour = st.slider("Hour of Day", 0, 23, 12)
month = st.slider("Month", 1, 12, 6)

# Compute renewables
renewables = solar + wind
st.write(f"🌱 Total Renewables: {renewables:.2f} MW")

# Create input DataFrame
input_data = pd.DataFrame([{
    'Solar': solar,
    'Wind': wind,
    'Renewables': renewables,
    'Imports': imports,
    'temp': temp,
    'humidity': humidity,
    'hour': hour,
    'month': month
}])

# Predict button
if st.button("🔍 Predict Electricity Load"):
    predicted_load = rf_best.predict(input_data)[0]
    st.success(f"⚡️ Predicted Electricity Load: {predicted_load:,.2f} MW")

    # Optional: show model evaluation
    y_pred = rf_best.predict(X_test)
    st.subheader("📊 Model Evaluation (on Test Set)")
    st.write("R² Score:", round(r2_score(y_test, y_pred), 4))
    st.write("Mean Squared Error:", round(mean_squared_error(y_test, y_pred), 2))
    st.write("Mean Absolute Error:", round(mean_absolute_error(y_test, y_pred), 2))
