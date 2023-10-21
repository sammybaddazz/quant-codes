#!/usr/bin/env python
# coding: utf-8

# In[2]:


from openbb_terminal.sdk import openbb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math


# In[6]:


def analyze_stock(ticker):
    data = openbb.stocks.load(ticker, start_date="2019-01-01", end_date="2023-10-01")
    
    # Calculate Exponential Moving Averages (EMAs)
    data['10_EMA'] = data['Close'].ewm(span=10).mean()
    data['50_EMA'] = data['Close'].ewm(span=50).mean()
    data['200_EMA'] = data['Close'].ewm(span=200).mean()
    data['3Y_EMA'] = data['Close'].ewm(span=3*252).mean()  # Assuming 252 trading days in a year
    
    # Calculate RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15,10), gridspec_kw={'height_ratios': [2, 1]})
    
    # Plotting
    ax1.plot(data.index, data['Close'], label='Price', color='green')
    ax1.plot(data.index, data["10_EMA"], label="10-Day EMA", color= 'orange')
    ax1.plot(data.index, data['50_EMA'], label='50-Day EMA', color='blue')
    ax1.plot(data.index, data['200_EMA'], label='200-Day EMA', color='red')
    ax1.plot(data.index, data['3Y_EMA'], label='3-Year EMA', color='magenta')  # 3-year exponential moving average

    ax1.set_title(f'Price and Exponential Moving Averages for {ticker}')
    ax1.set_ylabel('Price')
    ax1.legend(loc='best')
    
    # Plot the RSI on ax2
    ax2.plot(data.index, data['RSI'], label='RSI', color='green')
    ax2.axhline(70, color='red', linestyle='--')  # Overbought line
    ax2.axhline(30, color='blue', linestyle='--')  # Oversold line
    ax2.set_title(f'Relative Strength Index (RSI) for {ticker}')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('RSI')
    ax2.fill_between(data.index, y1=30, y2=70, color='grey', alpha=0.2)

    
    plt.tight_layout()
    plt.show()

# To analyze multiple stock tickers
tickers = ["DBC", "IAU", "EWJ", "EWZ", "SQM", "URA",]  # Add or remove tickers as needed
for ticker in tickers:
    analyze_stock(ticker)


# In[ ]:




