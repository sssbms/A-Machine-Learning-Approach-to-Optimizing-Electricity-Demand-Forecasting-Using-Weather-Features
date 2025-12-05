# === IMPORT LIBRARIES ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# === STEP 1: LOAD DATASET ===
df = pd.read_csv(r"C:\Users\User11\Downloads\ML ASSIGN\KD34403 ML PROJECT\KD34403 ML PROJECT\final_merged_objective2.csv")

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

# === STEP 7: DEFINE MODELS + TUNING ===
results = []

# --- Model 1: Ridge Regression (Tuned Linear Regression)
ridge_params = {
    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
}
ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=3, scoring='r2', n_jobs=-1)
ridge_grid.fit(X_train, y_train)
ridge_best = ridge_grid.best_estimator_
y_pred_ridge = ridge_best.predict(X_test)
results.append({
    'Model': 'Ridge Regression (Tuned)',
    'R²': round(r2_score(y_test, y_pred_ridge), 4),
    'MSE': round(mean_squared_error(y_test, y_pred_ridge), 2),
    'MAE': round(mean_absolute_error(y_test, y_pred_ridge), 2)
})

# --- Model 2: Random Forest with GridSearchCV
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
}
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=3, scoring='r2', n_jobs=-1)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_
y_pred_rf = rf_best.predict(X_test)
results.append({
    'Model': 'Random Forest (Tuned)',
    'R²': round(r2_score(y_test, y_pred_rf), 4),
    'MSE': round(mean_squared_error(y_test, y_pred_rf), 2),
    'MAE': round(mean_absolute_error(y_test, y_pred_rf), 2)
})

# --- Model 3: Gradient Boosting with GridSearchCV
gb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
}
gb_grid = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_params, cv=3, scoring='r2', n_jobs=-1)
gb_grid.fit(X_train, y_train)
gb_best = gb_grid.best_estimator_
y_pred_gb = gb_best.predict(X_test)
results.append({
    'Model': 'Gradient Boosting (Tuned)',
    'R²': round(r2_score(y_test, y_pred_gb), 4),
    'MSE': round(mean_squared_error(y_test, y_pred_gb), 2),
    'MAE': round(mean_absolute_error(y_test, y_pred_gb), 2)
})

# --- Model 4: XGBoost with GridSearchCV
xgb_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.9, 1.0]
}

xgb_grid = GridSearchCV(XGBRegressor(random_state=42, verbosity=0), xgb_params, cv=3, scoring='r2', n_jobs=-1)
xgb_grid.fit(X_train, y_train)
xgb_best = xgb_grid.best_estimator_
y_pred_xgb = xgb_best.predict(X_test)
results.append({
    'Model': 'XGBoost (Tuned)',
    'R²': round(r2_score(y_test, y_pred_xgb), 4),
    'MSE': round(mean_squared_error(y_test, y_pred_xgb), 2),
    'MAE': round(mean_absolute_error(y_test, y_pred_xgb), 2)
})

# === STEP 8: DISPLAY RESULTS ===
results_df = pd.DataFrame(results)
print("\n Model Evaluation Results (Tuned):\n")
print(results_df)

# === STEP 9: FEATURE IMPORTANCE PLOT FOR RANDOM FOREST ===
'''importances = rf_best.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='mako')
plt.title('Feature Importances - Random Forest (Tuned)')
plt.tight_layout()
plt.show()'''
