# === IMPORT LIBRARIES ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# === STEP 1: LOAD DATASET ===
df = pd.read_csv("C:\\Users\\User11\\Downloads\\ML ASSIGN\\KD34403 ML PROJECT\\KD34403 ML PROJECT\\final_merged_objective2.csv")


# === STEP 2: PROCESS DATETIME & CREATE NEW TIME FEATURES ===
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df['hour'] = df['datetime'].dt.hour
df['month'] = df['datetime'].dt.month
df['dayofweek'] = df['datetime'].dt.dayofweek

# === STEP 3: CREATE DERIVED FEATURE: RENEWABLES ===
df['Renewables'] = df['Solar'] + df['Wind']

# === STEP 4: DEFINE FEATURES AND TARGET ===
features = ['Solar', 'Wind', 'Renewables', 'Imports', 'temp', 'humidity', 'hour', 'month']
target = 'Load'

X = df[features].copy()
y = df[target]

# === STEP 5: HANDLE MISSING VALUES ===
X = X.fillna(X.mean())

# === STEP 6: TRAIN-TEST SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === STEP 7: DEFINE MODELS ===
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42, verbosity=0)
}

# === STEP 8: TRAIN & EVALUATE MODELS ===
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    results.append({
        'Model': name,
        'R²': round(r2, 4),
        'MSE': round(mse, 2),
        'MAE': round(mae, 2)
    })

# === STEP 9: DISPLAY RESULTS ===
results_df = pd.DataFrame(results)
print("\n Model Evaluation Results (Selected Features):\n")
print(results_df)

# === STEP 10: FEATURE IMPORTANCE (Optional for tree models) ===
tree_model = models['Random Forest']
importances = tree_model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot top features
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importances - Random Forest')
plt.tight_layout()
plt.show()
