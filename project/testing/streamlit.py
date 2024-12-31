import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import streamlit as st

# Streamlit UI
st.title("Stock Price Prediction")

# User inputs
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, GOOGL):", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2023-12-31"))
prediction_days = st.slider("Days to Predict", 1, 30, 7)

if ticker:
    # Fetch data
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data['Date'] = stock_data.index
    stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['MA200'] = stock_data['Close'].rolling(window=200).mean()

    # Display summary
    st.write(f"Data Summary for {ticker}")
    st.write(stock_data.describe())

    # Plot actual prices and moving averages
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stock_data['Date'], stock_data['Close'], label="Close Price")
    ax.plot(stock_data['Date'], stock_data['MA50'], label="MA50", linestyle='--')
    ax.plot(stock_data['Date'], stock_data['MA200'], label="MA200", linestyle='--')
    ax.set_title(f"{ticker} Stock Prices")
    ax.legend()
    st.pyplot(fig)

    # Feature engineering
    stock_data['Returns'] = stock_data['Close'].pct_change()
    stock_data['Lag_1'] = stock_data['Close'].shift(1)
    stock_data['Lag_2'] = stock_data['Close'].shift(2)

    # Prepare dataset
    stock_data.dropna(inplace=True)
    X = stock_data[['Lag_1', 'Lag_2', 'MA50', 'MA200']]
    y = stock_data['Close']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    st.write(f"Model RMSE: {rmse:.2f}")

    # Predict future prices
    last_row = stock_data.iloc[-1]
    future_prices = []
    for _ in range(prediction_days):
        new_row = np.array([[
            last_row['Lag_1'],
            last_row['Lag_2'],
            last_row['MA50'],
            last_row['MA200']
        ]]).reshape(1, -1)  # Ensure 2D shape
        future_price = model.predict(new_row)[0]
        future_prices.append(future_price)

        # Update lags
        last_row['Lag_2'] = last_row['Lag_1']
        last_row['Lag_1'] = future_price


    # Create future data
    future_dates = pd.date_range(stock_data['Date'].iloc[-1], periods=prediction_days + 1)[1:]
    future_data = pd.DataFrame({'Date': future_dates, 'Predicted_Price': future_prices})

    # Plot predicted prices
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stock_data['Date'], stock_data['Close'], label="Actual Prices")
    ax.plot(future_data['Date'], future_data['Predicted_Price'], label="Predicted Prices", linestyle='--')
    ax.set_title(f"{ticker} Stock Price Prediction")
    ax.legend()
    st.pyplot(fig)
