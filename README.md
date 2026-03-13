# Bike Sharing Demand – Regression Baseline

## Overview
This project builds a regression model to predict hourly bike rental demand based on weather and time information, using the **Bike Sharing Demand** competition dataset from Kaggle. The goal is to establish a solid baseline with classic ML models

## Data
- Source: Kaggle competition “Bike Sharing Demand”.
- Main features:
  - `datetime` (split into `hour`, `dayofweek`, `month`, `year`);
  - weather features: `temp`, `atemp`, `humidity`, `windspeed`;
  - categorical features: `season`, `holiday`, `workingday`, `weather`.
- Target: `count` – number of bike rentals per hour.

## Preprocessing
- Dropped `casual` and `registered` to avoid target leakage.
- Converted `datetime` to:
  - `hour`, `dayofweek`, `month`, `year`.
- Normalized continuous features:
  - `temp`, `atemp` divided by 10;
  - `humidity`, `windspeed` scaled by their max values.
- Train/test split:
  - 70% train, 30% test (`sample(frac=0.7, random_state=1)`).

## Models
I trained several regression models on the log‑transformed target `log1p(count)` and converted predictions back with `expm1`.

- Linear Regression (`sklearn.linear_model.LinearRegression`)
- Random Forest Regressor (`sklearn.ensemble.RandomForestRegressor`)
- Gradient Boosting Regressor (`sklearn.ensemble.GradientBoostingRegressor`)

## Key Findings
- Adding time‑based features derived from `datetime` (hour, day of week, month, year) significantly improves performance compared to using only raw weather variables.
- Tree‑based models (Random Forest and Gradient Boosting) achieve lower RMSLE than plain Linear Regression, which is expected for this kind of non‑linear problem. 
- Training on `log1p(count)` and evaluating with RMSLE produces more stable models and better leaderboard‑style scores than training on raw `count`.
- Deleting the least important features may reduce or not affect the prediction accuracy

## How to Run
1. Download the data from Kaggle:  
   https://www.kaggle.com/competitions/bike-sharing-demand/data
   or from GitHub
2. Install dependencies and code
3. Run the code
