import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from transformers import pipeline
import datetime
import requests

# --------------------------------------------------------------------------------
# 1. SETUP PAGE
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Crypto Price Prediction & Visualization",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💹 Crypto Price Prediction & Visualization App")
st.write(
    """
    This Streamlit application allows you to:
    1. **Fetch** and **Explore** historical cryptocurrency price data.
    2. **Engineer Features** using technical indicators.
    3. **Train** multiple machine learning models to **predict future prices**.
    4. **Compare** model performances.
    5. **Visualize** price trends and model predictions.
    6. **Predict** next price based on user inputs.
    7. Optionally, perform **Sentiment Analysis** on crypto-related text.
    """
)

# --------------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical data for the given ticker from Yahoo Finance.

    Args:
        ticker (str): The ticker symbol (e.g., 'BTC-USD' for Bitcoin).
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame containing historical price data.
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error("No data fetched. Please adjust the date range or ticker symbol.")
            st.stop()
        data.reset_index(inplace=True)  # Reset index to make 'Date' a column
        data.rename(columns={'Date': 'Timestamp'}, inplace=True)
        return data
    except Exception as e:
        st.error(f"An error occurred while fetching data: {e}")
        st.stop()

def flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flattens MultiIndex columns to single level by joining with an underscore.

    Args:
        df (pd.DataFrame): DataFrame with MultiIndex columns.

    Returns:
        pd.DataFrame: DataFrame with flattened columns.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds technical indicators to the DataFrame."""
    
    # Debug: Display DataFrame columns
    st.write("### DataFrame Columns Before Processing")
    st.write(df.columns.tolist())
    
    # Handle MultiIndex columns
    df = flatten_multiindex_columns(df)
    
    # Check for duplicate 'Close' columns
    close_columns = [col for col in df.columns if 'Close' in col]
    if len(close_columns) > 1:
        st.warning(f"Multiple 'Close' columns detected: {close_columns}. Using the first one.")
    
    # Select the appropriate 'Close' column
    if 'Close' in df.columns:
        df['Close'] = df['Close']
    elif 'Close Price' in df.columns:
        df['Close'] = df['Close Price']
    else:
        st.error("No 'Close' column found in the data.")
        st.stop()
    
    # Ensure 'Close' is a Series
    if isinstance(df['Close'], pd.DataFrame):
        st.warning("'Close' is a DataFrame with multiple columns. Selecting the first column.")
        df['Close'] = df['Close'].iloc[:, 0]
    
    # Moving Averages
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    df['RSI'] = 100 - (100 / (1 + RS))
    
    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window=20).std()
    
    # Drop rows with NaN values created by indicators
    df.dropna(inplace=True)
    
    # Debug: Display DataFrame columns after adding indicators
    st.write("### DataFrame Columns After Adding Technical Indicators")
    st.write(df.columns.tolist())
    
    return df

def train_models(df: pd.DataFrame):
    """
    Trains multiple regression models and returns a dictionary of trained models
    along with their performance metrics.

    Args:
        df (pd.DataFrame): DataFrame containing price data and technical indicators.

    Returns:
        trained_models (dict): Dictionary of trained models.
        X_test (np.array): Test features.
        y_test (np.array): Test target values.
        performance (dict): Performance metrics for each model.
    """
    # Create shifted feature for previous close price
    df['Close_lag1'] = df['Close'].shift(1)
    df = df.dropna()  # Drop first row with no lag value

    # Define feature columns
    feature_columns = ['Close_lag1', 'MA_10', 'MA_50', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'Volume']
    X = df[feature_columns].values
    y = df['Close'].values

    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Define models to train
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Support Vector Regressor': SVR(kernel='rbf')
    }

    trained_models = {}
    performance = {}

    # Train each model and evaluate
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        trained_models[name] = model
        performance[name] = {'MAE': mae, 'RMSE': rmse}

    return trained_models, X_test, y_test, performance

def fetch_real_time_data():
    """Fetches real-time BTC price data from CoinGecko API."""
    url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'
    params = {
        'vs_currency': 'usd',
        'days': '1',  # fetch data for the last day
        'interval': 'minutely'
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        prices = data['prices']  # list of [timestamp, price]
        df_real_time = pd.DataFrame(prices, columns=['Timestamp', 'Close'])
        df_real_time['Timestamp'] = pd.to_datetime(df_real_time['Timestamp'], unit='ms')
        return df_real_time
    except Exception as e:
        st.error(f"Error fetching real-time data: {e}")
        return pd.DataFrame()

# --------------------------------------------------------------------------------
# 3. USER INPUTS FOR DATA FETCHING
# --------------------------------------------------------------------------------

st.sidebar.header("🔍 Select Cryptocurrency and Date Range")

# List of available cryptocurrencies on Yahoo Finance (extend as needed)
available_tickers = {
    'Bitcoin': 'BTC-USD',
    'Ethereum': 'ETH-USD',
    'Ripple': 'XRP-USD',
    'Litecoin': 'LTC-USD',
    'Bitcoin Cash': 'BCH-USD'
}

selected_crypto = st.sidebar.selectbox("Select Cryptocurrency", list(available_tickers.keys()), index=0)
ticker_symbol = available_tickers[selected_crypto]

# Date selection
today = datetime.date.today()
default_start = today - datetime.timedelta(days=365*2)  # last 2 years
start_date = st.sidebar.date_input("Start Date", default_start)
end_date = st.sidebar.date_input("End Date", today)

if start_date > end_date:
    st.sidebar.error("Error: End date must fall after start date.")

# --------------------------------------------------------------------------------
# 4. LOAD DATA
# --------------------------------------------------------------------------------

df = load_data(ticker_symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

# --------------------------------------------------------------------------------
# 5. EXPLORATORY DATA ANALYSIS
# --------------------------------------------------------------------------------
st.subheader("📈 Exploratory Data Analysis")

# Set 'Timestamp' as index for plotting
if 'Timestamp' in df.columns:
    df.sort_values('Timestamp', inplace=True)
    df.set_index('Timestamp', inplace=True)

# Display data sample
st.write("### Data Sample")
st.dataframe(df.head())

# Display basic statistics
st.write("### Basic Statistics")
st.write(df.describe())

# Visualizations
# 1. Close Price Over Time
st.write("### Close Price Over Time")
fig_price = px.line(df.reset_index(), x='Timestamp', y='Close', title=f'{selected_crypto} Close Price Over Time')
st.plotly_chart(fig_price, use_container_width=True)

# 2. Correlation Heatmap
st.write("### Correlation Heatmap")
corr = df.corr()
fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu', title='Feature Correlation Heatmap')
st.plotly_chart(fig_corr, use_container_width=True)

# --------------------------------------------------------------------------------
# 6. ADD TECHNICAL INDICATORS
# --------------------------------------------------------------------------------
st.subheader("📊 Technical Indicators")

df = add_technical_indicators(df)

# Display technical indicators
st.write("### Data with Technical Indicators")
st.dataframe(df.head())

# --------------------------------------------------------------------------------
# 7. TRAIN PRICE PREDICTION MODELS
# --------------------------------------------------------------------------------
st.subheader("🧠 Train Price Prediction Models")

trained_models, X_test, y_test, model_performance = train_models(df.reset_index())

# --------------------------------------------------------------------------------
# 7.1 MODEL COMPARISON AND SELECTION
# --------------------------------------------------------------------------------

st.write("### Model Performance Metrics")
perf_df = pd.DataFrame(model_performance).T
perf_df = perf_df[['MAE', 'RMSE']]
perf_df = perf_df.rename(columns={'MAE': 'Mean Absolute Error (MAE)', 'RMSE': 'Root Mean Squared Error (RMSE)'})
st.dataframe(perf_df.style.highlight_min(axis=0, color='lightcoral').highlight_max(axis=0, color='lightgreen'))

# Allow user to select a model
st.write("### Select a Model for Prediction")
selected_model_name = st.selectbox("Choose Model", list(trained_models.keys()), index=0)
selected_model = trained_models[selected_model_name]

# Display selected model's performance
mae_selected = mean_absolute_error(y_test, selected_model.predict(X_test))
rmse_selected = mean_squared_error(y_test, selected_model.predict(X_test), squared=False)

st.write(f"**{selected_model_name} Performance:**")
col1, col2 = st.columns(2)
with col1:
    st.metric(label="MAE (Mean Absolute Error)", value=f"{mae_selected:.2f}")
with col2:
    st.metric(label="RMSE (Root Mean Squared Error)", value=f"{rmse_selected:.2f}")

# Plot Actual vs Predicted Close Price
st.write(f"### Actual vs Predicted Close Price - {selected_model_name}")
predictions = selected_model.predict(X_test)
fig_actual_pred = px.line(title=f'Actual vs Predicted Close Price - {selected_model_name}')
fig_actual_pred.add_scatter(y=y_test, mode='lines', name='Actual', line=dict(color='blue'))
fig_actual_pred.add_scatter(y=predictions, mode='lines', name='Predicted', line=dict(color='red'))
st.plotly_chart(fig_actual_pred, use_container_width=True)

# --------------------------------------------------------------------------------
# 8. PREDICT NEXT PRICE
# --------------------------------------------------------------------------------
st.subheader("🔮 Predict the Next Close Price")

# Get the latest available features
latest_features = df.iloc[-1][['Close_lag1', 'MA_10', 'MA_50', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'Volume']].values.reshape(1, -1)

# Allow user to override features for scenario testing
override = st.checkbox("🔧 Override Latest Features for Prediction")

if override:
    st.write("### Override Latest Features")
    override_features = []
    feature_names = ['Close_lag1', 'MA_10', 'MA_50', 'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'Volume']
    for i, feature in enumerate(feature_names):
        value = st.number_input(f"{feature}", value=float(latest_features[0][i]), format="%.2f")
        override_features.append(value)
    latest_features = np.array(override_features).reshape(1, -1)

# Make prediction
predicted_next_price = selected_model.predict(latest_features)[0]
st.write(f"**Predicted Next Close Price:** {predicted_next_price:.2f} USD")

# --------------------------------------------------------------------------------
# 9. SENTIMENT ANALYSIS
# --------------------------------------------------------------------------------
st.subheader("📝 Crypto News/Tweet Sentiment Analysis")

st.write("Enter any text related to cryptocurrency (e.g., news headlines, tweets) to analyze sentiment.")

# Load sentiment analysis pipeline
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis")

sentiment_analyzer = load_sentiment_pipeline()

# User input for sentiment analysis
user_text = st.text_area("Your Text Here", value="Bitcoin breaks new all-time high!")

if st.button("Analyze Sentiment"):
    if user_text.strip():
        sentiment_result = sentiment_analyzer(user_text)
        st.write("### Sentiment Result")
        st.write(sentiment_result)
    else:
        st.warning("Please enter some text to analyze.")

# --------------------------------------------------------------------------------
# 10. SENTIMENT ANALYSIS OVER TIME (SIMULATED DATA)
# --------------------------------------------------------------------------------
st.subheader("📊 Sentiment Analysis Over Time")

st.write(
    """
    *Note:* For demonstration purposes, this section uses simulated sentiment data.
    In a real-world application, you would fetch historical news or tweets data and perform sentiment analysis accordingly.
    """
)

# Simulate sentiment data
np.random.seed(42)
sentiments = np.random.choice(['POSITIVE', 'NEGATIVE'], size=len(df))
df['Sentiment'] = sentiments
df['Sentiment_Score'] = df['Sentiment'].apply(lambda x: 1 if x == 'POSITIVE' else -1)

# Plot Sentiment Score Over Time
fig_sentiment = px.line(df.reset_index(), x='Timestamp', y='Sentiment_Score',
                        title='Sentiment Score Over Time',
                        labels={'Sentiment_Score': 'Sentiment Score'})
st.plotly_chart(fig_sentiment, use_container_width=True)

# Correlate Sentiment with Close Price
fig_corr_sentiment = px.scatter(df, x='Sentiment_Score', y='Close', trendline='ols',
                                title='Sentiment Score vs. Close Price')
st.plotly_chart(fig_corr_sentiment, use_container_width=True)

# --------------------------------------------------------------------------------
# 11. REAL-TIME DATA FETCHING
# --------------------------------------------------------------------------------
st.subheader("⏱️ Real-Time BTC Price")

if st.button("Fetch Real-Time Data"):
    real_time_df = fetch_real_time_data()
    if not real_time_df.empty:
        st.write("### Real-Time Data")
        st.dataframe(real_time_df.head())
        
        # Plot Real-Time Close Price
        fig_rt = px.line(real_time_df, x='Timestamp', y='Close', title='Real-Time BTC Close Price')
        st.plotly_chart(fig_rt, use_container_width=True)
    else:
        st.warning("No real-time data available.")

# --------------------------------------------------------------------------------
# 12. FOOTER
# --------------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    **Developed by:** Your Name  
    **Data Sources:** [Yahoo Finance](https://finance.yahoo.com/), [CoinGecko API](https://www.coingecko.com/en/api)
    """
)
