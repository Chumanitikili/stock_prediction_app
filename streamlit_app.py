import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go
import time
import random
from pandas_ta import rsi
from datetime import datetime, timedelta
import requests

# Enhanced Market Tickers Dictionary
MARKET_TICKERS = {
    'US Stocks': {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
        'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
        'Healthcare': ['JNJ', 'PFE', 'UNH', 'MRK', 'ABT'],
        'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB']
    },
    'Forex': {
        'Major Pairs': ['EURUSD=X', 'USDJPY=X', 'GBPUSD=X', 'AUDUSD=X', 'USDCAD=X'],
        'Emerging Markets': ['USDCNY=X', 'USDINR=X', 'USDBRL=X', 'USDMXN=X', 'USDRUB=X']
    },
    'Indices': {
        'NASDAQ': ['^IXIC'],
        'US': ['^DJI', '^GSPC'],
        'Global': ['^N225', '^FTSE', '^GDAXI', '^HSI', '^BSESN']
    }
}

# Replace with your Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = "NZ8IP791ZRUHK4LL"  

def fetch_stock_data(ticker, start_date, end_date):
    min_days = 10  # Minimum days needed for preprocessing
    if (end_date - start_date).days < min_days:
        st.error(f"Date range must be at least {min_days} days. Please adjust your selection.")
        return None

    with st.spinner(f"Fetching data for {ticker}..."):
        for attempt in range(5):
            try:
                df = yf.download(ticker, start=start_date, end=end_date)
                df.reset_index(inplace=True)
                if df.empty or 'Date' not in df.columns or len(df) < min_days:
                    raise ValueError(f"Insufficient data returned for {ticker}")
                st.info(f"Successfully fetched {ticker} data from Yahoo Finance")
                return df
            except Exception as e:
                delay = min(2**attempt + random.uniform(0, 1), 10)
                st.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
        
        # Fallback to Alpha Vantage
        st.warning(f"Yahoo Finance failed for {ticker}. Trying Alpha Vantage...")
        try:
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}&outputsize=full"
            response = requests.get(url, timeout=10)
            data = response.json()
            if "Time Series (Daily)" not in data:
                raise ValueError("Invalid response from Alpha Vantage")
            df = pd.DataFrame(data["Time Series (Daily)"]).T
            df = df.rename(columns={"1. open": "Open", "2. high": "High", "3. low": "Low", "4. close": "Close", "5. volume": "Volume"})
            df.index = pd.to_datetime(df.index)
            df = df.astype(float)
            df = df.reset_index().rename(columns={"index": "Date"})
            df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
            if len(df) < min_days:
                raise ValueError(f"Insufficient data from Alpha Vantage for {ticker}")
            st.info(f"Successfully fetched {ticker} data from Alpha Vantage")
            return df
        except Exception as e:
            st.error(f"Alpha Vantage failed: {e}. Using demo data.")
            df = pd.DataFrame({
                'Date': pd.date_range(start=start_date, periods=max(200, (end_date - start_date).days), freq='D'),
                'Open': np.random.rand(max(200, (end_date - start_date).days)) * 100 + 100,
                'High': np.random.rand(max(200, (end_date - start_date).days)) * 100 + 105,
                'Low': np.random.rand(max(200, (end_date - start_date).days)) * 100 + 95,
                'Close': np.random.rand(max(200, (end_date - start_date).days)) * 100 + 100,
                'Volume': np.random.randint(1000, 10000, max(200, (end_date - start_date).days))
            })
            return df

def prepare_data(df, look_back=5):
    if df is None or len(df) < look_back + 1:
        st.error(f"Insufficient raw data for processing (need at least {look_back + 1} days)")
        return None, None, None
    
    # Check for NaN in 'Close'
    if df['Close'].hasnans:
        st.error("Close price has NaN values. Please ensure data is complete.")
        return None, None, None
    
    try:
        # Add technical indicators with fill to minimize NaN loss
        df['MA3'] = df['Close'].rolling(window=3, min_periods=1).mean().bfill()  # Use bfill() instead of fillna(method='bfill')
        df['MA5'] = df['Close'].rolling(window=5, min_periods=1).mean().bfill()  # Use bfill() instead of fillna(method='bfill')
        
        # Calculate RSI
        rsi_series = rsi(df['Close'], length=5)
        if rsi_series is None:
            st.error("RSI calculation failed. Please check the input data.")
            return None, None, None
        
        df['RSI'] = rsi_series.bfill()  # Use bfill() instead of fillna(method='bfill')
    except Exception as e:
        st.error(f"Error calculating technical indicators: {e}")
        return None, None, None
    
    if len(df) < 2:
        st.error("Not enough data after preprocessing")
        return None, None, None
    
    # Prepare features
    features = ['Close', 'MA3', 'MA5', 'RSI']
    data = df[features].values
    
    # Normalize
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i])
        y.append(scaled_data[i, 0])  # Predict 'Close'
    
    X, y = np.array(X), np.array(y)
    return X, y, scaler

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def plot_predictions(df, predictions, scaler, ticker):
    actual_prices = df['Close'].values[-len(predictions):].reshape(-1, 1)
    pred_array = np.zeros((len(predictions), 4))  # Match scaler input shape
    pred_array[:, 0] = predictions.flatten()
    predictions = scaler.inverse_transform(pred_array)[:, 0]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'][-len(predictions):], y=actual_prices.flatten(), mode='lines', name='Actual Price'))
    fig.add_trace(go.Scatter(x=df['Date'][-len(predictions):], y=predictions, mode='lines', name='Predicted Price', line=dict(dash='dash')))
    fig.update_layout(title=f"{ticker} Historical Prediction", xaxis_title="Date", yaxis_title="Price", legend=dict(x=0, y=1))
    st.plotly_chart(fig)

def main():
    st.title('Financial Market Prediction App')
    
    # Market and Category Selection
    market = st.selectbox('Select Market', list(MARKET_TICKERS.keys()))
    category = st.selectbox('Select Category', list(MARKET_TICKERS[market].keys()))
    ticker = st.selectbox('Select Ticker', MARKET_TICKERS[market][category])
    
    # Date Range with validation
    start_date = st.date_input("Start Date", value=datetime(2022, 1, 1))  # Extended to ensure more data
    end_date = st.date_input("End Date", value=datetime(2023, 12, 31))
    
    # Prediction Horizon
    days_ahead = st.slider("Days to Predict Ahead", 1, 60, 30)
    
    if st.button('Predict Prices'):
        df = fetch_stock_data(ticker, start_date, end_date)
        
        if df is not None:
            # Prepare data
            X, y, scaler = prepare_data(df)
            
            if X is not None and y is not None:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train LSTM
                model = create_lstm_model(input_shape=(X.shape[1], X.shape[2]))
                with st.spinner("Training model..."):
                    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)
                
                # Make predictions
                predictions = model.predict(X_test)
                
                # Plot historical predictions
                st.subheader("Historical Prediction vs Actual")
                plot_predictions(df, predictions, scaler, ticker)
                
                # Future predictions
                last_sequence = X[-1:]
                future_preds = []
                for _ in range(days_ahead):
                    next_pred = model.predict(last_sequence, verbose=0)
                    future_preds.append(next_pred[0, 0])
                    last_sequence = np.roll(last_sequence, -1, axis=1)
                    last_sequence[0, -1] = np.append(next_pred, last_sequence[0, -1, 1:]).reshape(1, -1)
                
                future_dates = pd.date_range(start=df['Date'].max() + timedelta(days=1), periods=days_ahead, freq='D')
                future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': scaler.inverse_transform(np.array(future_preds).reshape(-1, 4))[:, 0]})
                
                st.subheader("Future Price Prediction")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Historical'))
                fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted_Close'], mode='lines', name='Predicted', line=dict(dash='dash')))
                fig.update_layout(title=f"{ticker} Future Prediction", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig)

if __name__ == '__main__':
    main()