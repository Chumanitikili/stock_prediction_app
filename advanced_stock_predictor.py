import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, concatenate
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go
import time
import random
from pandas_ta import rsi, macd, bbands, stoch, adx
from datetime import datetime, timedelta
import requests
import ta
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
import warnings
warnings.filterwarnings('ignore')

# Enhanced Market Tickers Dictionary
MARKET_TICKERS = {
    'US Stocks': {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'],
        'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA'],
        'Healthcare': ['JNJ', 'PFE', 'UNH', 'MRK', 'ABT', 'ABBV', 'AMGN'],
        'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'BP', 'SHELL'],
        'Consumer': ['WMT', 'PG', 'KO', 'MCD', 'NKE', 'DIS', 'NFLX']
    },
    'Forex': {
        'Major Pairs': ['EURUSD=X', 'USDJPY=X', 'GBPUSD=X', 'AUDUSD=X', 'USDCAD=X', 'NZDUSD=X', 'USDCHF=X'],
        'Emerging Markets': ['USDCNY=X', 'USDINR=X', 'USDBRL=X', 'USDMXN=X', 'USDRUB=X', 'USDKRW=X', 'USDZAR=X'],
        'Cross Pairs': ['EURGBP=X', 'EURJPY=X', 'GBPJPY=X', 'AUDJPY=X', 'CADJPY=X']
    },
    'Crypto': {
        'Major': ['bitcoin', 'ethereum', 'binancecoin', 'ripple', 'cardano', 'solana', 'dogecoin'],
        'DeFi': ['uniswap', 'aave', 'compound-governance-token', 'maker', 'havven'],
        'Layer 2': ['matic-network', 'optimism', 'arbitrum', 'immutable-x']
    },
    'Indices': {
        'US': ['^DJI', '^GSPC', '^IXIC', '^RUT', '^VIX'],
        'Global': ['^N225', '^FTSE', '^GDAXI', '^HSI', '^BSESN', '^FCHI', '^STOXX50E'],
        'Emerging': ['^NSEI', '^SSEC', '^KOSPI', '^BVSP']
    },
    'Commodities': {
        'Precious Metals': ['GC=F', 'SI=F', 'PL=F', 'PA=F'],
        'Energy': ['CL=F', 'NG=F', 'RB=F', 'HO=F'],
        'Agriculture': ['ZC=F', 'ZW=F', 'KC=F', 'CT=F']
    }
}

# API Keys
ALPHA_VANTAGE_API_KEY = "NZ8IP791ZRUHK4LL"

def fetch_crypto_data(coin_id, start_date, end_date):
    """Fetch cryptocurrency data from CoinGecko API"""
    try:
        # Convert dates to Unix timestamps
        start_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        end_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())
        
        # CoinGecko API endpoint
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
        params = {
            'vs_currency': 'usd',
            'from': start_timestamp,
            'to': end_timestamp
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data['prices'], columns=['Date', 'Close'])
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        df.set_index('Date', inplace=True)
        
        # Add OHLCV data
        df['Open'] = df['Close'].shift(1)
        df['High'] = df[['Open', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'Close']].min(axis=1)
        df['Volume'] = 0  # CoinGecko free API doesn't provide volume data
        
        # Reset index to make Date a column
        df.reset_index(inplace=True)
        
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
        # Try multiple data sources
        for attempt in range(5):
            try:
                # Check if it's a crypto ticker
                if ticker in [coin for category in MARKET_TICKERS['Crypto'].values() for coin in category]:
                    df = fetch_crypto_data(ticker, start_date, end_date)
                else:
                    df = yf.download(ticker, start=start_date, end=end_date)
                    df.reset_index(inplace=True)
                
                if df is None or df.empty or 'Date' not in df.columns or len(df) < min_days:
                    raise ValueError(f"Insufficient data for {ticker}")
                
                # Add more technical indicators
                df['MA3'] = df['Close'].rolling(window=3, min_periods=1).mean()
                df['MA5'] = df['Close'].rolling(window=5, min_periods=1).mean()
                df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
                df['MA50'] = df['Close'].rolling(window=50, min_periods=1).mean()
                
                # RSI
                rsi_indicator = RSIIndicator(close=df['Close'])
                df['RSI'] = rsi_indicator.rsi()
                
                # MACD
                macd_indicator = ta.trend.MACD(close=df['Close'])
                df['MACD'] = macd_indicator.macd()
                df['MACD_Signal'] = macd_indicator.macd_signal()
                
                # Bollinger Bands
                bollinger = BollingerBands(close=df['Close'])
                df['BB_High'] = bollinger.bollinger_hband()
                df['BB_Low'] = bollinger.bollinger_lband()
                df['BB_Middle'] = bollinger.bollinger_mavg()
                
                # Stochastic Oscillator
                stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
                df['Stoch_K'] = stoch.stoch()
                df['Stoch_D'] = stoch.stoch_signal()
                
                # ADX
                adx_indicator = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'])
                df['ADX'] = adx_indicator.adx()
                
                # Volume indicators
                df['Volume_MA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
                df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
                
                # Price momentum
                df['Price_Change'] = df['Close'].pct_change()
                df['Price_Volatility'] = df['Price_Change'].rolling(window=20, min_periods=1).std()
                
                st.info(f"Fetched {ticker} data with technical indicators")
                return df
                
            except Exception as e:
                delay = min(2**attempt + random.uniform(0, 1), 10)
                st.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
                time.sleep(delay)
        
        st.error(f"Failed to fetch data for {ticker} after multiple attempts")
        return None

def create_advanced_model(input_shape):
    # Price input branch
    price_input = Input(shape=input_shape, name='price_input')
    price_lstm1 = LSTM(100, return_sequences=True)(price_input)
    price_lstm1 = Dropout(0.3)(price_lstm1)
    price_lstm2 = LSTM(50)(price_lstm1)
    price_lstm2 = Dropout(0.3)(price_lstm2)
    
    # Technical indicators input branch
    tech_input = Input(shape=input_shape, name='tech_input')
    tech_lstm1 = LSTM(100, return_sequences=True)(tech_input)
    tech_lstm1 = Dropout(0.3)(tech_lstm1)
    tech_lstm2 = LSTM(50)(tech_lstm1)
    tech_lstm2 = Dropout(0.3)(tech_lstm2)
    
    # Combine branches
    combined = concatenate([price_lstm2, tech_lstm2])
    dense1 = Dense(100, activation='relu')(combined)
    dense1 = Dropout(0.3)(dense1)
    dense2 = Dense(50, activation='relu')(dense1)
    dense2 = Dropout(0.3)(dense2)
    output = Dense(1)(dense2)
    
    model = Model(inputs=[price_input, tech_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def prepare_data(df, look_back=5):
    if df is None or len(df) < look_back + 1:
        st.error(f"Need at least {look_back + 1} days of data")
        return None, None, None, None
    
    # Price features
    price_features = ['Close', 'Open', 'High', 'Low', 'Volume']
    price_data = df[price_features].values
    
    # Technical indicators
    tech_features = ['MA3', 'MA5', 'MA20', 'MA50', 'RSI', 'MACD', 'BB_High', 'BB_Low', 
                    'Stoch_K', 'Stoch_D', 'ADX', 'Volume_Ratio', 'Price_Volatility']
    tech_data = df[tech_features].values
    
    # Scale the data
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    tech_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_price = price_scaler.fit_transform(price_data)
    scaled_tech = tech_scaler.fit_transform(tech_data)
    
    X_price, X_tech, y = [], [], []
    for i in range(look_back, len(scaled_price)):
        X_price.append(scaled_price[i-look_back:i])
        X_tech.append(scaled_tech[i-look_back:i])
        y.append(scaled_price[i, 0])
    
    X_price = np.array(X_price)
    X_tech = np.array(X_tech)
    y = np.array(y)
    
    return X_price, X_tech, y, (price_scaler, tech_scaler)

def plot_predictions(df, predictions, scalers, ticker):
    price_scaler, tech_scaler = scalers
    actual_prices = df['Close'].values[-len(predictions):]
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatter(
        x=df['Date'][-len(predictions):],
        y=actual_prices,
        name='Actual',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Date'][-len(predictions):],
        y=predictions,
        name='Predicted',
        line=dict(color='red', dash='dash')
    ))
    
    # Add RSI
    fig.add_trace(go.Scatter(
        x=df['Date'][-len(predictions):],
        y=df['RSI'][-len(predictions):],
        name='RSI',
        yaxis='y2'
    ))
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} Price Prediction with RSI",
        xaxis_title="Date",
        yaxis_title="Price",
        yaxis2=dict(
            title="RSI",
            overlaying="y",
            side="right",
            range=[0, 100]
        )
    )
    
    st.plotly_chart(fig)

def main():
    st.set_page_config(page_title="Advanced Financial Market Prediction", layout="wide")
    st.title('Advanced Financial Market Prediction App')
    
    # Sidebar for market selection
    with st.sidebar:
        st.header("Market Selection")
        market = st.selectbox('Select Market', list(MARKET_TICKERS.keys()))
        category = st.selectbox('Select Category', list(MARKET_TICKERS[market].keys()))
        ticker = st.selectbox('Select Ticker', MARKET_TICKERS[market][category])
        
        st.header("Analysis Parameters")
        start_date = st.date_input("Start Date", value=datetime(2022, 1, 1))
        end_date = st.date_input("End Date", value=datetime(2023, 12, 31))
        days_ahead = st.slider("Days to Predict Ahead", 1, 60, 30)
        look_back = st.slider("Look Back Period", 5, 30, 10)
        
        st.header("Model Parameters")
        epochs = st.slider("Training Epochs", 10, 100, 20)
        batch_size = st.slider("Batch Size", 16, 128, 32)
    
    if st.button('Generate Predictions'):
        df = fetch_stock_data(ticker, start_date, end_date)
        
        if df is not None:
            # Display market overview
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
            with col2:
                price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100
                st.metric("24h Change", f"{price_change:.2f}%")
            with col3:
                st.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")
            
            # Prepare data and train model
            X_price, X_tech, y, scalers = prepare_data(df, look_back)
            
            if X_price is not None and X_tech is not None:
                X_price_train, X_price_test, X_tech_train, X_tech_test, y_train, y_test = train_test_split(
                    X_price, X_tech, y, test_size=0.2, random_state=42
                )
                
                model = create_advanced_model(input_shape=(X_price.shape[1], X_price.shape[2]))
                
                with st.spinner("Training advanced model..."):
                    history = model.fit(
                        [X_price_train, X_tech_train],
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.2,
                        verbose=0
                    )
                
                # Generate predictions
                predictions = model.predict([X_price_test, X_tech_test])
                st.subheader("Historical Prediction vs Actual")
                plot_predictions(df, predictions, scalers, ticker)
                
                # Generate future predictions
                last_price_sequence = X_price[-1:]
                last_tech_sequence = X_tech[-1:]
                future_preds = []
                
                for _ in range(days_ahead):
                    next_pred = model.predict([last_price_sequence, last_tech_sequence], verbose=0)
                    future_preds.append(next_pred[0, 0])
                    
                    # Update sequences for next prediction
                    last_price_sequence = np.roll(last_price_sequence, -1, axis=1)
                    last_price_sequence[0, -1] = np.append(next_pred, last_price_sequence[0, -1, 1:]).reshape(1, -1)
                    
                    last_tech_sequence = np.roll(last_tech_sequence, -1, axis=1)
                    last_tech_sequence[0, -1] = last_tech_sequence[0, -1]
                
                # Plot future predictions
                future_dates = pd.date_range(start=df['Date'].max() + timedelta(days=1), periods=days_ahead, freq='D')
                future_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted_Close': scalers[0].inverse_transform(
                        np.array(future_preds).reshape(-1, 5)
                    )[:, 0]
                })
                
                st.subheader("Future Price Prediction")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Historical'))
                fig.add_trace(go.Scatter(
                    x=future_df['Date'],
                    y=future_df['Predicted_Close'],
                    mode='lines',
                    name='Predicted',
                    line=dict(dash='dash')
                ))
                fig.update_layout(
                    title=f"{ticker} Future Prediction",
                    xaxis_title="Date",
                    yaxis_title="Price"
                )
                st.plotly_chart(fig)
                
                # Display prediction metrics
                st.subheader("Model Performance Metrics")
                mse = np.mean((predictions - y_test) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(predictions - y_test))
                st.write(f"Mean Squared Error: {mse:.4f}")
                st.write(f"Root Mean Squared Error: {rmse:.4f}")
                st.write(f"Mean Absolute Error: {mae:.4f}")

if __name__ == '__main__':
    main() 