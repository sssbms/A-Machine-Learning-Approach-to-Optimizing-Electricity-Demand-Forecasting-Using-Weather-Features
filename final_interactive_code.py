# === IMPORT LIBRARIES ===
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# === LOAD & PREPARE DATA ===
df = pd.read_csv("C:\\Users\\User11\\Downloads\\ML ASSIGN\\KD34403 ML PROJECT\\KD34403 ML PROJECT\\final_merged_objective2.csv")
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df['hour'] = df['datetime'].dt.hour
df['month'] = df['datetime'].dt.month
df['dayofweek'] = df['datetime'].dt.dayofweek
df['Renewables'] = df['Solar'] + df['Wind']

# === FEATURES & TARGET ===
features = ['Solar', 'Wind', 'Renewables', 'Imports', 'temp', 'humidity', 'hour', 'month']
target = 'Load'
X = df[features].copy()
y = df[target]
X = X.fillna(X.mean())

# === TRAIN-TEST SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === RANDOM FOREST TUNING ===
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
}
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=3, scoring='r2', n_jobs=-1)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_

# === MODEL EVALUATION ===
y_pred = rf_best.predict(X_test)
print("\n Model Evaluation (Final Random Forest):")
print("R²:", round(r2_score(y_test, y_pred), 4))
print("MSE:", round(mean_squared_error(y_test, y_pred), 2))
print("MAE:", round(mean_absolute_error(y_test, y_pred), 2))

# === INTERACTIVE PREDICTION FUNCTION ===
def interactive_predict():
    print("\n Enter input values to predict electricity demand:")
    solar = float(input("Enter Solar Generation (MW): "))
    wind = float(input("Enter Wind Generation (MW): "))
    imports = float(input("Enter Imports (MW): "))
    temp = float(input("Enter Temperature (°C): "))
    humidity = float(input("Enter Humidity (%): "))
    hour = int(input("Enter Hour (0-23): "))
    month = int(input("Enter Month (1-12): "))
    
    renewables = solar + wind
    print(f" Total Renewables (Solar + Wind): {renewables} MW")  # ← Add this line
    
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
    
    predicted_load = rf_best.predict(input_data)[0]
    print(f"\n Predicted Electricity Load: {predicted_load:,.2f} MW")

# === RUN INTERACTIVE LOOP ===
while True:
    interactive_predict()
    again = input("\n Predict again? (yes/no): ").strip().lower()
    if again != "yes":
        print(" Exiting prediction tool. Goodbye!")
        break
