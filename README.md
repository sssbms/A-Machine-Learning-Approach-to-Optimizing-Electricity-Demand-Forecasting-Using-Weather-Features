# A-Machine-Learning-Approach-to-Optimizing-Electricity-Demand-Forecasting-Using-Weather-Features

This repository presents a machine learning approach to predicting electricity demand using energy production and weather data. The project evaluates multiple regression models, compares baseline and tuned performance, and implements a prototype application for real-time demand prediction.

# Objectives

•Develop an electricity demand prediction model using energy and weather features
• Improve prediction performance through model tuning and evaluation
• Create an interactive prototype that allows users to input variables and generate demand predictions

# Dataset Description

The project uses three datasets covering the year 2022:

• Curtailment data: Solar and wind energy curtailments recorded at 5-minute intervals
• Production data: Electricity demand and generation details across different energy sources
• Weather data: Hourly environmental readings including temperature, humidity, wind, rainfall, solar radiation, pressure, and visibility

All datasets were cleaned, merged using a datetime key, and stored as a unified file for modelling.

# Pre-processing Summary

• Created standardized datetime fields
• Removed duplicates and consolidated curtailment intervals
• Merged datasets and filled missing values
• Dropped derived or irrelevant features to avoid leakage
• Final merged dataset (hourly records) was saved for modelling

# Model Development

Four regression algorithms were trained:

• Linear Regression
• Random Forest
• Gradient Boosting
• XGBoost

Models were evaluated using R², MSE, and MAE. Hyperparameter optimisation was applied to improve performance.

Baseline Model Performance

Random Forest achieved the best accuracy before tuning, followed by XGBoost. Linear Regression performed the worst, suggesting it could not capture nonlinear relationships between electricity demand and weather variables.

# Tuned Model Performance

After tuning:

• Random Forest remained the most accurate model
• XGBoost improved slightly
• Gradient Boosting also showed notable improvement
• Linear Regression performance remained unchanged

The tuning results confirmed the effectiveness of ensemble-based machine learning for forecasting demand under weather influence.

# Prototype Application

A prototype was developed using Streamlit to demonstrate real-time prediction capability.
Users input:
• Solar power (MW)
• Wind power (MW)
• Imports (MW)
• Temperature
• Humidity
• Hour
• Month

The system predicts electricity demand instantly. This interface was designed for operational use in power plants, supporting decision-making for supply planning and system management.

# Conclusion
This project shows that integrating weather and energy datasets with machine learning improves electricity demand forecasting accuracy. Ensemble models, especially Random Forest and XGBoost, significantly outperform traditional linear approaches. The prototype demonstrates practical application potential for industry adoption. Future work includes simplifying inputs, enhancing interpretability, and adapting models to new regional or temporal contexts.
