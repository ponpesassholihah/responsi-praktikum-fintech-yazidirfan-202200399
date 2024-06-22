import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Set title
st.title('LSTM Stock Price Prediction')

# Sidebar for ticker input
ticker = st.sidebar.text_input('Stock Ticker', 'AAPL')

# Load data
@st.cache
def load_data(ticker):
    data = yf.download(ticker, start='2010-01-01', end='2023-01-01')
    return data

data = load_data(ticker)

# Show raw data
st.subheader('Raw Data')
st.write(data.tail())

# Preprocessing
data = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Prepare data for LSTM
train_data_len = int(np.ceil(len(scaled_data) * 0.8))
train_data = scaled_data[0:int(train_data_len), :]

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Test data
test_data = scaled_data[train_data_len - 60:, :]
x_test = []
y_test = data[train_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Plot the data
train = data[:train_data_len]
valid = data[train_data_len:]
valid = np.append(valid[:0], predictions)

st.subheader('Predicted vs Actual Closing Price')
fig, ax = plt.subplots()
ax.plot(train, label='Train Data')
ax.plot(valid, label='Validation Data')
ax.legend()
st.pyplot(fig)
