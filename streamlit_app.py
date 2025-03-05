import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objs as go

def download_stock_data(ticker, start_date, end_date):
    """
    Download stock data using yfinance with error handling.
    """
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        
        # Check if data is empty
        if df.empty:
            st.error(f"No data available for {ticker} between {start_date} and {end_date}")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error downloading stock data: {e}")
        return None

def prepare_data(df, look_back=60):
    """
    Prepare data for LSTM model with improved error handling.
    """
    # Check if dataframe is None or empty
    if df is None or df.empty:
        st.error("No data available to prepare")
        return None, None, None
    
    # Drop rows with NaN values and reset index
    df_cleaned = df.dropna().reset_index()
    
    # Check if there's enough data after cleaning
    if len(df_cleaned) < look_back + 1:
        st.error(f"Need at least {look_back + 1} days of data")
        return None, None, None
    
    # Use Close price for prediction
    close_prices = df_cleaned['Close'].values
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(close_prices.reshape(-1, 1))
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_prices) - look_back):
        X.append(scaled_prices[i:i+look_back])
        y.append(scaled_prices[i+look_back])
    
    X, y = np.array(X), np.array(y)
    
    return X, y, scaler

def create_lstm_model(input_shape):
    """
    Create LSTM model for stock price prediction.
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def plot_predictions(actual, predicted, scaler):
    """
    Create a Plotly chart comparing actual and predicted prices.
    """
    # Inverse transform the scaled values
    actual_prices = scaler.inverse_transform(actual)
    predicted_prices = scaler.inverse_transform(predicted)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=actual_prices.flatten(), mode='lines', name='Actual Prices'))
    fig.add_trace(go.Scatter(y=predicted_prices.flatten(), mode='lines', name='Predicted Prices'))
    
    fig.update_layout(
        title='Stock Price Prediction',
        xaxis_title='Time',
        yaxis_title='Price',
        hovermode='x'
    )
    
    return fig

def main():
    st.title('Stock Price Prediction App')
    
    # Sidebar for input
    st.sidebar.header('Stock Prediction Parameters')
    ticker = st.sidebar.text_input('Enter Stock Ticker', 'AAPL')
    start_date = st.sidebar.date_input('Start Date', pd.Timestamp.now() - pd.Timedelta(days=365))
    end_date = st.sidebar.date_input('End Date', pd.Timestamp.now())
    
    # Download stock data
    df = download_stock_data(ticker, start_date, end_date)
    
    if df is not None:
        st.subheader("Stock Data")
        st.write(df)
        
        # Prepare data
        X, y, scaler = prepare_data(df)
        
        if X is not None and y is not None:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Reshape input for LSTM [samples, time steps, features]
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            # Create and train model
            model = create_lstm_model(input_shape=(X_train.shape[1], 1))
            history = model.fit(
                X_train, y_train, 
                epochs=50, 
                batch_size=32, 
                validation_split=0.2, 
                verbose=0
            )
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Plot results
            fig = plot_predictions(y_test, y_pred, scaler)
            st.plotly_chart(fig)
            
            # Display model performance
            st.subheader('Model Performance')
            st.write(f'Test Loss: {model.evaluate(X_test, y_test, verbose=0)}')

if __name__ == '__main__':
    main()