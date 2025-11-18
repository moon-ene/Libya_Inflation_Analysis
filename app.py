import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set page title

st.set_page_config(page_title="Libya GDP Forecast ", layout="wide")
st.title("Libya GDP Forecast ")

# Initialize FRED API
api_key = "1b5e76d94e9dc29c08bc917cd9074004"
fred = Fred(api_key=api_key)

# Fetch Libya GDP data
st.write("Fetching Libya GDP data from FRED...")
libya = fred.get_series("LBYNGDPDUSD")
libya_df = libya.to_frame(name="GDP_USD")
libya_df = libya_df.reset_index()
libya_df.rename(columns={'index': 'ds'}, inplace=True)
libya_df["ds"] = pd.to_datetime(libya_df["ds"])


# Resample to quarterly frequency and interpolate
libya_df.set_index('ds', inplace=True)
libya_df_quarterly = libya_df.resample('QE').mean().interpolate(method='linear')
libya_df_quarterly.reset_index(inplace=True)
libya_df_quarterly.rename(columns={"GDP_USD": "y"}, inplace=True)


# Prepare the series for SARIMA
gdp_series = libya_df_quarterly.set_index("ds")["y"]
gdp_series = gdp_series.asfreq('Q')



# Define SARIMA parameter ranges for grid search
p = d = q = range(0, 3)
P = D = Q = range(0, 2)
s = 4  # Quarterly seasonality

pdq = list(itertools.product(p, d, q))
seasonal_pdq = list(itertools.product(P, D, Q, [s]))

# Grid search for best SARIMA parameters
best_aic = float('inf')
best_params = None
best_model = None

st.write("Performing grid search for SARIMA parameters (as in notebook)...")

for param in pdq:
    for seasonal_param in seasonal_pdq:
        try:
            model = SARIMAX(gdp_series[:'2025-03-31'],
                            order=param,
                            seasonal_order=seasonal_param,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            results = model.fit(disp=False)
            if results.aic < best_aic:
                best_aic = results.aic
                best_params = (param, seasonal_param)
                best_model = results
        except Exception as e:
            continue  # optionally print error for debugging

# Unpack best parameters
(p, d, q), (P, D, Q, s) = best_params

st.write(f"Best SARIMA parameters: order=({p},{d},{q}), seasonal_order=({P},{D},{Q},{s})")
st.write(f"Best AIC: {best_aic:.2f}")
# Evaluate on test set (2021-2025)
pred_test = best_model.get_prediction(start='2021-03-31', end='2025-03-31')
pred_mean_test = pred_test.predicted_mean
actual_test = gdp_series['2021-03-31':'2025-03-31']
mae = mean_absolute_error(actual_test, pred_mean_test)
rmse = np.sqrt(mean_squared_error(actual_test, pred_mean_test))
mape = np.mean(np.abs((actual_test - pred_mean_test) / actual_test)) * 100

st.write(f"SARIMA Test MAE: {mae:.2f}")
st.write(f"SARIMA Test RMSE: {rmse:.2f}")
st.write(f"SARIMA Test MAPE: {mape:.2f}%")

# Forecast 24 quarters (2025Q2-2031Q1)
forecast_steps = 24
forecast = best_model.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start='2025-06-30', periods=forecast_steps, freq='Q')
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()
forecast_mean.index = forecast_index
forecast_ci.index = forecast_index

# Plotting
st.write("Generating forecast plot...")
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(gdp_series[:'2025-03-31'], label='Historical GDP', color='b')
ax.plot(gdp_series['2021-03-31':'2025-03-31'], label='Actual GDP (Test)', color='c')
ax.plot(pred_mean_test, label='Forecasted GDP (Test)', linestyle='--', color='r')
ax.plot(forecast_mean, label='Forecasted GDP (SARIMA)', linestyle='--', color='r', alpha=0.5)
ax.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], 
                color='orange', alpha=0.3, label='Confidence Interval')
ax.set_title("SARIMA Forecast: Libya Quarterly GDP")
ax.set_xlabel("Date")
ax.set_ylabel("GDP (USD)")
ax.legend()
ax.grid(True)
ax.tick_params(axis='x', rotation=45)
fig.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig)




