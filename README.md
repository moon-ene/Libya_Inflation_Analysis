# Libya_Inflation_Analysis

# Libya GDP Analysis & Forecasting

This repository contains an end-to-end analysis and forecasting project focused on Libya’s Gross Domestic Product (GDP).
The goal is to analyze historical GDP patterns, build predictive models, and generate insights to support economic monitoring and policy analysis.

Project Overview

Libya’s GDP has historically shown high volatility due to geopolitical instability, oil-driven revenue cycles, and production disruptions.
This notebook performs:

Data acquisition from the Federal Reserve Economic Data (FRED) API

Exploratory Data Analysis (EDA) on long-term GDP trends

Time-series preprocessing, transformations, and stationarity checks

Modeling using:

ARIMA / SARIMAX

Prophet

LSTM deep learning networks

Model comparison using MAE, RMSE, and forecast visualizations

Future GDP forecasting and insights

Main Tools & Libraries

pandas, numpy

matplotlib, seaborn, plotly

fredapi

statsmodels (ARIMA / SARIMAX)

prophet

tensorflow / keras

scikit-learn

Workflow Summary
1. Data Collection

GDP data for Libya is retrieved via FRED using an API key.

2. Data Cleaning & EDA

Trend visualization

Checking seasonality

Handling missing values

Rolling averages and smoothing

3. Stationarity Testing

ADF test

Differencing and transformation where needed

4. Modeling

The notebook implements several forecasting models:

ARIMA / SARIMAX

Classical statistical models effective for trend-based forecasting.

Prophet

Robust to irregular time intervals and suitable for economic seasonality trends.

LSTM Neural Network

Sequence-based deep learning architecture including:

Timesteps windowing

MinMax scaling

Sequential Keras model with LSTM layers

5. Model Evaluation

Models are compared using:

MAE

MSE / RMSE

Forecast accuracy plots

Side-by-side visual forecasts

6. Future Forecasting

The notebook generates multi-year GDP forecasts and plots long-term projection curves for Libya.
