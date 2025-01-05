import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import pipeline
import datetime

# --------------------------------------------------------------------------------
# 1. SETUP PAGE
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Crypto Sentiment & Price Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Crypto Sentiment & Price Prediction App")
st.write(
    """
    This Streamlit application demonstrates a simple workflow of:
    1. Loading and exploring historical Bitcoin price data.
    2. Training a linear regression model to predict Bitcoin prices.
    3. Using a Hugging Face model for sentiment analysis of crypto-related text.
    """
)

# --------------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------------------------------------

@st.cache_data
def load_data(url_or_path: str) -> pd.DataFrame:
    """Loads the CSV data from a URL or local path into a pandas DataFrame."""
    df = pd.read_csv(url_or_path)
    return df

def train_model(df: pd.DataFrame):
    """
    Trains a simple linear regression model based on provided DataFrame.
    Returns the trained model, X_test, y_test for evaluation.
    """
    # For simplicity, let's use 'Close' price at time t-1 to predict 'Close' price at time t
    # We'll create a shifted feature
    df['Close_lag1'] = df['Close'].shift(1)
    df = df.dropna()  # drop first row with no lag value

    X = df[['Close_lag1']].values
    y = df['Close'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_test, y_test

# --------------------------------------------------------------------------------
# 3. LOAD DATA
# --------------------------------------------------------------------------------

# You can replace the link below with your own dataset or local file path.
# This is an example dataset from Kaggle or any other open source for BTC historical data.
# Example: https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data
DATA_URL = "https://raw.githubusercontent.com/fivat-lab/crypto-datasets/main/BTC_2015-2021_kaggle.csv"

df = load_data(DATA_URL)

# Convert timestamp column to datetime if needed; 
# adjust column names in your real dataset if they differ
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
elif 'Date' in df.columns:
    # If there's a 'Date' column, parse it as datetime
    df['Date'] = pd.to_datetime(df['Date'])

# For demonstration, let's rename 'Close' column to ensure uniform naming
if 'Close' not in df.columns:
    if 'Close Price' in df.columns:
        df.rename(columns={'Close Price': 'Close'}, inplace=True)

# Filter columns we care about
expected_columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume_(Currency)']
df = df[[col for col in expected_columns if col in df.columns]]

# Remove duplicates or nulls
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# --------------------------------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS
# --------------------------------------------------------------------------------
st.subheader("Exploratory Data Analysis")

if 'Timestamp' in df.columns:
    df.sort_values('Timestamp', inplace=True)
    df.set_index('Timestamp', inplace=True)  # set datetime as index for plotting

st.write("### Data Sample")
st.dataframe(df.head())

# Basic stats
st.write("### Basic Statistics")
st.write(df.describe())

# Line chart of Close price
st.write("### Close Price Over Time")
fig, ax = plt.subplots(figsize=(10,4))
sns.lineplot(x=df.index, y='Close', data=df, ax=ax)
ax.set_xlabel("Date")
ax.set_ylabel("BTC Close Price")
st.pyplot(fig)

# --------------------------------------------------------------------------------
# 5. TRAIN PRICE PREDICTION MODEL
# --------------------------------------------------------------------------------
model, X_test, y_test = train_model(df.reset_index())

# Evaluate model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)

st.subheader("Linear Regression Model Performance")
col1, col2 = st.columns(2)
with col1:
    st.metric(label="MAE (Mean Absolute Error)", value=f"{mae:.2f}")
with col2:
    st.metric(label="RMSE (Root Mean Squared Error)", value=f"{rmse:.2f}")

# Plot actual vs. predicted
st.write("### Actual vs. Predicted Close Price (Test Set)")
fig2, ax2 = plt.subplots(figsize=(10,4))
ax2.plot(y_test, label="Actual", color='blue')
ax2.plot(predictions, label="Predicted", color='red')
ax2.set_xlabel("Test Data Index")
ax2.set_ylabel("Close Price")
ax2.legend()
st.pyplot(fig2)

# --------------------------------------------------------------------------------
# 6. FORECAST NEXT PRICE
# --------------------------------------------------------------------------------
st.subheader("Predict the Next BTC Price")

# Next price is basically model.predict(last_close_price)
last_close_price = df['Close'].iloc[-1]
user_input = st.number_input("If you want to override the last known BTC Close price (for scenario testing), enter a value:",
                             min_value=0.0, value=float(last_close_price))

predicted_price = model.predict(np.array(user_input).reshape(-1,1))[0]
st.write(f"**Predicted Next Price**: {predicted_price:.2f} USD")

# --------------------------------------------------------------------------------
# 7. HUGGING FACE SENTIMENT ANALYSIS
# --------------------------------------------------------------------------------
st.subheader("Crypto News/Tweet Sentiment Analysis")
st.write("Enter any text (crypto news headlines, tweets, etc.) to analyze sentiment.")

# Load the Hugging Face sentiment pipeline
# By default: "distilbert-base-uncased-finetuned-sst-2-english"
sentiment_analyzer = pipeline("sentiment-analysis")

user_text = st.text_area("Your text here", value="Bitcoin breaks new all-time high!")

if user_text.strip():
    sentiment_result = sentiment_analyzer(user_text)
    st.write("### Sentiment Result")
    st.write(sentiment_result)