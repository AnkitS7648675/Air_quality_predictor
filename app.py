import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly

# Load and preprocess your data
df = pd.read_csv('air_pollution_data.csv')
df = df.rename(columns={'date': 'ds', 'aqi': 'y'})
df['ds'] = pd.to_datetime(df['ds'])

# Fit model
model = Prophet()
model.fit(df)

# Forecast
future = model.make_future_dataframe(periods=7)
forecast = model.predict(future)

# Streamlit UI
st.title("Air Quality Forecast")
st.write("Forecast data:")
st.dataframe(forecast[['ds', 'yhat']].tail(7))

# Plot
fig = plot_plotly(model, forecast)
st.plotly_chart(fig)
