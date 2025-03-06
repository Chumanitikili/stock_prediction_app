import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, concatenate
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go
import time
import random
from ta.trend import MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.trend import ADXIndicator
from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings('ignore')

# Enhanced Market Tickers Dictionary
MARKET_TICKERS = {
    'US Stocks': {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'],
        'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA']
    },
    'Forex': {
        'Major Pairs': ['EURUSD=X', 'USDJPY=X', 'GBPUSD=X', 'AUDUSD=X', 'USDCAD=X']
    },
    'Crypto': {
        'Major': ['bitcoin', 'ethereum', 'binancecoin', 'ripple', 'cardano']
    },
    'Indices': {
        'US': ['^DJI', '^GSPC', '^IXIC']
    }
}

ALPHA_VANTAGE_API_KEY = "NZ8IP791ZRUHK4LL"

def fetch_crypto_data(coin_id, start_date, end_date):
    try:
        start_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        end_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
        params = {'vs_currency': 'usd', 'from': start_timestamp, 'to': end_timestamp}
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data['prices'], columns=['Date', 'Close'])
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        df['Open'] = df['Close'].shift(1).fillna(df['Close'])  # Approximate Open
        df['High'] = df[['Open', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'Close']].min(axis=1)
        df['Volume'] = 1  # Default volume for crypto
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching crypto data: {str(e)}")
        return None

def fetch_stock_data(ticker, start_date, end_date):
    min_days = 10
    if (end_date - start_date).days < min_days:
        st.error(f"Date range must be at least {min_days} days.")
        return None

    with st.spinner(f"Fetching data for {ticker}..."):
        for attempt in range(3):
            try:
                is_crypto = ticker in [coin for category in MARKET_TICKERS['Crypto'].values() for coin in category]
                if is_crypto:
                    df = fetch_crypto_data(ticker, start_date, end_date)
                else:
                    df = yf.download(ticker, start=start_date, end=end_date)
                    df.reset_index(inplace=True)
                
                if df is None or df.empty or 'Date' not in df.columns or len(df) < min_days:
                    raise ValueError(f"Insufficient data for {ticker}")
                
                # Add technical indicators with NaN handling
                df['MA3'] = df['Close'].rolling(window=3, min_periods=1).mean().ffill().bfill()
                df['MA5'] = df['Close'].rolling(window=5, min_periods=1).mean().ffill().bfill()
                df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean().ffill().bfill()
                
                rsi_indicator = RSIIndicator(close=df['Close'], window=14)
                df['RSI'] = rsi_indicator.rsi().ffill().bfill()
                
                macd_indicator = MACD(close=df['Close'])
                df['MACD'] = macd_indicator.macd().ffill().bfill()
                
                bollinger = BollingerBands(close=df['Close'])
                df['BB_High'] = bollinger.bollinger_hband().ffill().bfill()
                df['BB_Low'] = bollinger.bollinger_lband().ffill().bfill()
                
                stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
                df['Stoch_K'] = stoch.stoch().ffill().bfill()
                
                adx_indicator = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'])
                df['ADX'] = adx_indicator.adx().ffill().bfill()
                
                df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20, min_periods=1).mean().ffill().bfill()
                df['Price_Volatility'] = df['Close'].pct_change().rolling(window=20, min_periods=1).std().ffill().bfill()
                
                # Drop any remaining NaN rows
                df = df.dropna()
                
                st.info(f"Fetched {ticker} data with indicators")
                return df
            except Exception as e:
                delay = min(2**attempt + random.uniform(0, 1), 5)
                st.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
                time.sleep(delay)
        st.error(f"Failed to fetch data for {ticker}")
        return None

def create_advanced_model(input_shape):
    price_input = Input(shape=input_shape, name='price_input')
    price_lstm1 = LSTM(100, return_sequences=True)(price_input)
    price_lstm1 = Dropout(0.3)(price_lstm1)
    price_lstm2 = LSTM(50)(price_lstm1)
    price_lstm2 = Dropout(0.3)(price_lstm2)
    
    tech_input = Input(shape=input_shape, name='tech_input')
    tech_lstm1 = LSTM(100, return_sequences=True)(tech_input)
    tech_lstm1 = Dropout(0.3)(tech_lstm1)
    tech_lstm2 = LSTM(50)(tech_lstm1)
    tech_lstm2 = Dropout(0.3)(tech_lstm2)
    
    combined = concatenate([price_lstm2, tech_lstm2])
    dense1 = Dense(100, activation='relu')(combined)
    dense1 = Dropout(0.3)(dense1)
    output = Dense(1)(dense1)
    
    model = Model(inputs=[price_input, tech_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def prepare_data(df, look_back=5):
    if df is None or len(df) < look_back + 1:
        st.error(f"Need at least {look_back + 1} days of data")
        return None, None, None, None
    
    price_features = ['Close', 'Open', 'High', 'Low', 'Volume']
    tech_features = ['MA3', 'MA5', 'MA20', 'RSI', 'MACD', 'BB_High', 'BB_Low', 'Stoch_K', 'ADX', 'Volume_Ratio', 'Price_Volatility']
    
    price_data = df[price_features].values
    tech_data = df[tech_features].values
    
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    tech_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_price = price_scaler.fit_transform(price_data)
    scaled_tech = tech_scaler.fit_transform(tech_data)
    
    X_price, X_tech, y = [], [], []
    for i in range(look_back, len(scaled_price)):
        X_price.append(scaled_price[i-look_back:i])
        X_tech.append(scaled_tech[i-look_back:i])
        y.append(scaled_price[i, 0])  # Predict Close
    
    X_price, X_tech, y = np.array(X_price), np.array(X_tech), np.array(y)
    return X_price, X_tech, y, (price_scaler, tech_scaler)

def plot_predictions(df, predictions, scaler, ticker):
    actual_prices = df['Close'].values[-len(predictions):]
    pred_array = np.zeros((len(predictions), 5))
    pred_array[:, 0] = predictions.flatten()
    predictions = scaler.inverse_transform(pred_array)[:, 0]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'][-len(predictions):], y=actual_prices, name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['Date'][-len(predictions):], y=predictions, name='Predicted', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=df['Date'][-len(predictions):], y=df['RSI'][-len(predictions):], name='RSI', yaxis='y2'))
    fig.update_layout(
        title=f"{ticker} Prediction with RSI",
        xaxis_title="Date",
        yaxis_title="Price",
        yaxis2=dict(title="RSI", overlaying="y", side="right", range=[0, 100])
    )
    st.plotly_chart(fig)

def main():
    st.set_page_config(page_title="Advanced Financial Prediction", layout="wide")
    st.title('Advanced Financial Market Prediction App')
    
    with st.sidebar:
        st.header("Market Selection")
        market = st.selectbox('Select Market', list(MARKET_TICKERS.keys()))
        category = st.selectbox('Select Category', list(MARKET_TICKERS[market].keys()))
        ticker = st.selectbox('Select Ticker', MARKET_TICKERS[market][category])
        
        st.header("Parameters")
        start_date = st.date_input("Start Date", value=datetime(2022, 1, 1))
        end_date = st.date_input("End Date", value=datetime(2023, 12, 31))
        days_ahead = st.slider("Days Ahead", 1, 60, 30)
        look_back = st.slider("Look Back", 5, 30, 10)
        epochs = st.slider("Epochs", 10, 100, 20)
        batch_size = st.slider("Batch Size", 16, 128, 32)
    
    if st.button('Generate Predictions'):
        df = fetch_stock_data(ticker, start_date, end_date)
        
        if df is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
            with col2:
                price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                st.metric("24h Change", f"{price_change:.2f}%")
            with col3:
                st.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")
            
            X_price, X_tech, y, scalers = prepare_data(df, look_back)
            if X_price is not None and X_tech is not None and y is not None:
                X_price_train, X_price_test, X_tech_train, X_tech_test, y_train, y_test = train_test_split(
                    X_price, X_tech, y, test_size=0.2, random_state=42
                )
                
                model = create_advanced_model((look_back, X_price.shape[2]))
                with st.spinner("Training model..."):
                    model.fit(
                        [X_price_train, X_tech_train], y_train,
                        epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0
                    )
                
                predictions = model.predict([X_price_test, X_tech_test])
                st.subheader("Historical Prediction")
                plot_predictions(df, predictions, scalers[0], ticker)
                
                # Future predictions
                last_price_seq = X_price[-1:]
                last_tech_seq = X_tech[-1:]
                future_preds = []
                for _ in range(days_ahead):
                    next_pred = model.predict([last_price_seq, last_tech_seq], verbose=0)[0, 0]
                    future_preds.append(next_pred)
                    
                    # Update price sequence
                    new_price = np.zeros((1, 1, 5))
                    new_price[0, 0, 0] = next_pred  # Update Close
                    last_price_seq = np.concatenate([last_price_seq[:, 1:, :], new_price], axis=1)
                    
                    # Update tech sequence with approximations
                    new_tech = last_tech_seq[:, -1:, :].copy()  # Repeat last tech values
                    last_tech_seq = np.concatenate([last_tech_seq[:, 1:, :], new_tech], axis=1)
                
                future_dates = pd.date_range(start=df['Date'].max() + timedelta(days=1), periods=days_ahead, freq='D')
                future_preds_unscaled = scalers[0].inverse_transform(
                    np.array(future_preds).reshape(-1, 1).repeat(5, axis=1)
                )[:, 0]
                future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_preds_unscaled})
                
                st.subheader("Future Prediction")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Historical'))
                fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted_Close'], mode='lines', name='Predicted', line=dict(dash='dash')))
                fig.update_layout(title=f"{ticker} Future Prediction", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig)
                
                # Metrics
                mse = np.mean((predictions - y_test) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(predictions - y_test))
                st.subheader("Performance Metrics")
                st.write(f"MSE: {mse:.4f}")
                st.write(f"RMSE: {rmse:.4f}")
                st.write(f"MAE: {mae:.4f}")

if __name__ == '__main__':
    main()