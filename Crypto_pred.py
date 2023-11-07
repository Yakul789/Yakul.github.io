import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import pandas as pd
import datetime
from prophet import Prophet
import streamlit as st
from prophet.plot import plot_plotly
from plotly import graph_objs as go

#interval of forecasting(From 1st Jan 2020 to now)
start = '2020-01-01'
current = datetime.datetime.now()

#structure for web application
st.title('Cryptocurrency Forecasts')

# Mapping Tickers to Crypto currencies
#crypto_mapping = {
 #   'BTC-USD': 'Bitcoin',
  #  'ETH-USD': 'Ethereum',
   # 'DOGE-USD': 'Dogecoin',
    #'BNB-USD': 'Binance Coin',
#    'ADA-USD': 'Cardano',
 #   'SOL-USD': 'Solana',
  #  'XRP-USD': 'Ripple',
   # 'DOT-USD': 'Polkadot',
#    'LTC-USD': 'Litecoin',
 #   'BCH-USD': 'Bitcoin Cash'
#}

# Reverse the dictionary to map names to tickers
#reverse_mapping = {v: k for k, v in crypto_mapping.items()}
# Use the selected name to fetch the corresponding cryptocurrency ticker
#selected_ticker = crypto[selected_name]

#Show available cryptocurrencies
crypto = ('BTC-USD','ETH-USD','DOGE-USD','BNB-USD','ADA-USD','SOL-USD','XRP-USD','DOT-USD','LTC-USD','BCH-USD')

# Use the names as the values for the selectbox options
selected_ticker = st.selectbox('Select Cryptocurrency', crypto)

n_months = st.slider('Month of Forecast', 1, 5)  # Slider for weeks of forecast
period = n_months * 30

# Define the data loading function using the selected ticker
@st.cache_data  # Cache the searched data
def data_load(crypto):
    data = yf.download(selected_ticker, start, current)
    data.reset_index(inplace=True)
    return data

# Plot the current data in table format
data = data_load(selected_ticker)
st.subheader('Current trend')
st.write(data.tail())


#plot current trend of closing prices
candlestick_trace = go.Candlestick(
    x=data['Date'],
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close']
)
#barplot for plotting current trade volume data
volume_trace = go.Bar(
    x=data['Date'],
    y=data['Volume'],
    name='Volume',
    marker=dict(color='blue')  # Customize bar color
)

#use declared functions to plot the visualizations
def plot_current():
    fig_1 = go.Figure(data=[candlestick_trace])
    fig_1.layout.update(title_text='Closing Price Trend', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig_1)

    fig_2 = go.Figure(data=[volume_trace])
    fig_2.layout.update(title_text='Trade Volume Trend', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig_2)
#call the function
plot_current()

#prepare train data for forecast
#use the date and close columns
#we will try forecasting closing prices 
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={'Date': 'ds', 'Close': 'y'})

#we will use the prophet model
#create an object of prophet class and feed the training data to it
mod = Prophet()
mod.fit(df_train)
future = mod.make_future_dataframe(periods=period)#makes the future forecasted dataframe 
forecast = mod.predict(future)#makes the predictions

#plot forecasted data in dataframe format
st.subheader('Forecast data')
st.write(forecast.tail())

#plot line plot for forecast
fig1 = plot_plotly(mod, forecast)
st.plotly_chart(fig1)

#plot weekly and yearly trend components
st.write('Forecast components')
fig2 = mod.plot_components(forecast)
st.write(fig2)






