import yfinance as yf
import streamlit as st
import pandas as pd
import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.write("""
# Simple Stock Price App

Shown are the stock **closing price** and **volume**, along with LSTM predictions.
""")

def user_input_features():
    stock_symbol = st.sidebar.selectbox('Symbol', ('BMRI.JK', 'APLN.JK', 'MNCN.JK', 'BFIN.JK', 'CSAP.JK'))
    date_start = st.sidebar.date_input("Start Date", datetime.date(2015, 5, 31))
    date_end = st.sidebar.date_input("End Date", datetime.date.today())

    tickerData = yf.Ticker(stock_symbol)
    tickerDf = tickerData.history(period='1d', start=date_start, end=date_end)
    return tickerDf

input_df = user_input_features()

# Normalisasi data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(input_df['Close'].values.reshape(-1,1))

# Membuat data pelatihan
training_data_len = int(np.ceil( len(scaled_data) * .95 ))

train_data = scaled_data[0:int(training_data_len), :]

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape data
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# Membangun model LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Melatih model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Membuat data testing
test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = input_df['Close'][training_data_len:].values

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)

# Reshape data
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

# Membuat prediksi
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Menampilkan grafik harga penutupan
st.line_chart(input_df['Close'])
# Menampilkan grafik volume perdagangan
st.line_chart(input_df['Volume'])

# Menampilkan prediksi LSTM
input_df['Predictions'] = np.nan
input_df['Predictions'][training_data_len:] = predictions.flatten()
st.line_chart(input_df[['Close', 'Predictions']])
