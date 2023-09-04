#!/usr/bin/env python
# coding: utf-8

# In[3]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the ticker symbol
tickerSymbol = '^GSPC'

# Get data on this ticker
tickerData = yf.Ticker(tickerSymbol)

# Get the historical prices for this ticker
data = tickerData.history(period='1d', start='2000-01-01', end='2017-12-31')

# Calculate 250-day TMA
def triangular_moving_average(data, window_size):
    tma_values = []
    for i in range(len(data)):
        if i < window_size - 1:
            tma_values.append(np.nan)
        else:
            weights = np.arange(1, window_size + 1)
            tma = np.sum(data['Close'][i - window_size + 1 : i + 1] * weights) / np.sum(weights)
            tma_values.append(tma)
    return tma_values

window_size = 250
data['TMA'] = triangular_moving_average(data, window_size)

# Drop NaN values
data = data.dropna()

# Classify prices into regimes
regime_colors = np.where(data['Close'] > data['TMA'], 'green', 'red')

# Create a plot
plt.figure(figsize=(14, 7))

# Plot TMA
plt.plot(data.index, data['TMA'], color='blue', label='TMA')

# Color-coded lines based on regimes
for regime, color in [('Bullish', 'green'), ('Bearish', 'red')]:
    mask = regime_colors == color
    plt.plot(data.index[mask], data['Close'][mask], color=color, label=f'Price ({regime})')

plt.title('Price with TMA-based Coloring')
plt.ylabel('Price')
plt.xlabel('Date')
plt.legend()
plt.grid(True, color='white')
plt.gca().set_facecolor('grey')

plt.tight_layout()
plt.show()


# In[ ]:




