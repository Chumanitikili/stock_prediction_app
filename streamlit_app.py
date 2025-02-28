import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
import time

try:
    import streamlit as st
    IN_STREAMLIT = True
except ImportError:
    IN_STREAMLIT = False
    from IPython.display import display

class StockPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.data = None
        
    def fetch_data(self, ticker, start_date, end_date):
        """Fetch stock data from Yahoo Finance with retry"""
        for _ in range(3):  # Retry up to 3 times
            try:
                self.data = yf.download(ticker, start=start_date, end=end_date)
                self.data.reset_index(inplace=True)
                if self.data.empty or 'Date' not in self.data.columns:
                    raise ValueError("No valid data returned")
                if IN_STREAMLIT:
                    st.success(f"Successfully fetched data for {ticker}")
                else:
                    print(f"Successfully fetched data for {ticker}")
                return True
            except Exception as e:
                if IN_STREAMLIT:
                    st.warning(f"Retry attempt due to error: {e}")
                else:
                    print(f"Retry attempt due to error: {e}")
                time.sleep(1)  # Wait 1 second before retry
        if IN_STREAMLIT:
            st.error(f"Failed to fetch data for {ticker} after retries")
        else:
            print(f"Failed to fetch data for {ticker} after retries")
        return False
    
    def prepare_features(self):
        """Prepare features for prediction"""
        if self.data is None or self.data.empty:
            if IN_STREAMLIT:
                st.error("No data available to process")
            else:
                print("Error: No data available to process")
            return None, None
        
        self.data['Date'] = pd.to_datetime(self.data['Date'])  # Ensure datetime
        self.data['MA5'] = self.data['Close'].rolling(window=5).mean()
        self.data['MA20'] = self.data['Close'].rolling(window=20).mean()
        self.data['Days'] = (self.data['Date'] - self.data['Date'].min()).dt.days
        self.data = self.data.dropna()
        self.features = ['Days', 'MA5', 'MA20', 'Open', 'High', 'Low', 'Volume']
        self.target = 'Close'
        return self.data[self.features], self.data[self.target]
    
    def train_model(self):
        """Train the prediction model"""
        X, y = self.prepare_features()
        if X is None or len(X) < 2:
            if IN_STREAMLIT:
                st.error("Not enough data to train the model after preprocessing")
            else:
                print("Error: Not enough data to train the model after preprocessing")
            return None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        metrics = {
            'train_score': r2_score(y_train, train_pred),
            'test_score': r2_score(y_test, test_pred),
            'train_mse': mean_squared_error(y_train, train_pred),
            'test_mse': mean_squared_error(y_test, test_pred),
        }
        return metrics
    
    def predict_future(self, days_ahead=30):
        """Predict future stock prices"""
        last_date = self.data['Date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead, freq='D')
        
        last_valid_row = self.data[self.features].dropna().iloc[-1]
        future_data = pd.DataFrame({
            'Days': (future_dates - self.data['Date'].min()).days,
            'MA5': [float(last_valid_row['MA5'].iloc[0])] * days_ahead,
            'MA20': [float(last_valid_row['MA20'].iloc[0])] * days_ahead,
            'Open': [float(last_valid_row['Open'].iloc[0])] * days_ahead,
            'High': [float(last_valid_row['High'].iloc[0])] * days_ahead,
            'Low': [float(last_valid_row['Low'].iloc[0])] * days_ahead,
            'Volume': [float(last_valid_row['Volume'].iloc[0])] * days_ahead
        })
        
        future_predictions = self.model.predict(future_data[self.features])
        if future_predictions.ndim > 1:
            future_predictions = future_predictions.flatten()
        
        return pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_predictions})

# Main execution
predictor = StockPredictor()

if IN_STREAMLIT:
    st.title("Stock Price Prediction App")
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
    start_date = st.date_input("Start Date", value=datetime(2003, 1, 1))
    end_date = st.date_input("End Date", value=datetime(2023, 12, 31))
    days_ahead = st.slider("Days to Predict Ahead", 1, 60, 30)
    
    if st.button("Analyze Stock"):
        if predictor.fetch_data(ticker, start_date, end_date):
            st.subheader(f"{ticker} Historical Data")
            st.write(predictor.data.tail())
            metrics = predictor.train_model()
            if metrics:
                st.subheader("Model Performance")
                st.write(f"Training R² Score: {metrics['train_score']:.4f}")
                st.write(f"Test R² Score: {metrics['test_score']:.4f}")
                st.write(f"Training MSE: {metrics['train_mse']:.2f}")
                st.write(f"Test MSE: {metrics['test_mse']:.2f}")
                future_df = predictor.predict_future(days_ahead)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=predictor.data['Date'], y=predictor.data['Close'], mode='lines', name='Historical'))
                fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted_Close'], mode='lines', name='Predicted', line=dict(dash='dash')))
                fig.update_layout(title=f"{ticker} Stock Price Prediction", xaxis_title="Date", yaxis_title="Price", legend=dict(x=0, y=1))
                st.plotly_chart(fig)
else:
    ticker = "AAPL"
    start_date = datetime(2003, 1, 1)
    end_date = datetime(2023, 12, 31)
    days_ahead = 30
    
    if predictor.fetch_data(ticker, start_date, end_date):
        print(f"{ticker} Historical Data (Last 5 rows):")
        display(predictor.data.tail())
        metrics = predictor.train_model()
        if metrics:
            print("Model Performance:")
            print(f"Training R² Score: {metrics['train_score']:.4f}")
            print(f"Test R² Score: {metrics['test_score']:.4f}")
            print(f"Training MSE: {metrics['train_mse']:.2f}")
            print(f"Test MSE: {metrics['test_mse']:.2f}")
            future_df = predictor.predict_future(days_ahead)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=predictor.data['Date'], y=predictor.data['Close'], mode='lines', name='Historical'))
            fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted_Close'], mode='lines', name='Predicted', line=dict(dash='dash')))
            fig.update_layout(title=f"{ticker} Stock Price Prediction", xaxis_title="Date", yaxis_title="Price", legend=dict(x=0, y=1))
            fig.show()