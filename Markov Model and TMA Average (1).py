#!/usr/bin/env python
# coding: utf-8

# In[34]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Define the ticker symbol
tickerSymbol = '^GSPC'

# Get data on this ticker
tickerData = yf.Ticker(tickerSymbol)

# Get the historical prices for this ticker
data = tickerData.history(period='1d', start='2000-01-01', end='2017-12-31')

# Calculate 250-period TMA
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

window_size = 250  # Use the desired window size
data['TMA'] = triangular_moving_average(data, window_size)

# Compute daily returns
data['Returns'] = data['Close'].pct_change()

# Drop NaN values from returns and TMA
data = data.dropna()

# Fit the model
model = sm.tsa.MarkovAutoregression(data['Returns'], k_regimes=2, order=1, switching_variance=True)
res = model.fit()

# Get smoothed probabilities
data['Smoothed Probability 0'] = res.smoothed_marginal_probabilities[0]
data['Smoothed Probability 1'] = res.smoothed_marginal_probabilities[1]

# Create two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

# Plot the prices with color-coded lines based on regimes
ax1.plot(data.index, data['Close'], color='gray', linestyle='dashed', label='Price')
for regime, color in [(0, 'blue'), (1, 'green')]:
    mask = data['Smoothed Probability 0' if regime == 0 else 'Smoothed Probability 1'] >= 0.5
    ax1.plot(data.index[mask], data['Close'][mask], color=color, label=f'Regime {regime}')
ax1.set(title='Price with Regime-based Coloring',
       ylabel='Price')
ax1.legend()
ax1.grid(True, color='white')
ax1.set_facecolor('grey')

# Plot the prices with color-coded lines based on TMA
ax2.plot(data.index, data['TMA'], color='orange', label='TMA')
ax2.plot(data.index, data['Close'], color='gray', linestyle='dashed', label='Price')
tma_colors = np.where(data['Close'] > data['TMA'], 'green', 'red')
for regime, color in [(0, 'green'), (1, 'red')]:
    mask = tma_colors == color
    ax2.plot(data.index[mask], data['Close'][mask], color=color, label=f'Price ({"Bullish" if regime == 0 else "Bearish"})')
ax2.set(title='Price with TMA-based Coloring',
       ylabel='Price',
       xlabel='Date')
ax2.legend()
ax2.grid(True, color='white')
ax2.set_facecolor('grey')

plt.tight_layout()
plt.show()

# Print the model parameters and summary
print(res.params)
print(res.summary())


# In[ ]:




