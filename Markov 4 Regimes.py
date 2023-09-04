#!/usr/bin/env python
# coding: utf-8

# In[27]:


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

# Compute daily returns
data['Returns'] = data['Close'].pct_change()

# Drop NaN values from returns
data = data.dropna()

# Fit the model
model = sm.tsa.MarkovAutoregression(data['Returns'], k_regimes=2, order=1, switching_variance=True)
res = model.fit()

# Get smoothed probabilities
data['Smoothed Probability 0'] = res.smoothed_marginal_probabilities[0]
data['Smoothed Probability 1'] = res.smoothed_marginal_probabilities[1]

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
data['Regime'] = np.where(data['Close'] > data['TMA'], 'Bullish', 'Bearish')

# Define a function to classify regimes
def classify_regime(row):
    if row['Regime'] == 'Bullish' and row['Smoothed Probability 1'] >= 0.5:
        return 'Bullish with high variance'
    if row['Regime'] == 'Bullish' and row['Smoothed Probability 0'] >= 0.5:
        return 'Bullish with low variance'
    if row['Regime'] == 'Bearish' and row['Smoothed Probability 1'] >= 0.5:
        return 'Bearish with high variance'
    if row['Regime'] == 'Bearish' and row['Smoothed Probability 0'] >= 0.5:
        return 'Bearish with low variance'

# Apply the function to each row
data['Regime Classification'] = data.apply(classify_regime, axis=1)

# Plot the results
plt.figure(figsize=(20, 14))


# Color-coded lines based on regimes
regime_colors = {
    'Bullish with high variance': 'green',
    'Bullish with low variance': 'purple',
    'Bearish with high variance': 'blue',
    'Bearish with low variance': 'red',
}

for regime, color in regime_colors.items():
    mask = data['Regime Classification'] == regime
    plt.plot(data.index[mask], data['Close'][mask], color=color, label=f'Price ({regime})')

plt.title('Price with Combined Regime-based Coloring')
plt.ylabel('Price')
plt.xlabel('Date')
plt.legend()
plt.grid(True, color='white')
plt.gca().set_facecolor('grey')

plt.tight_layout()
plt.show()


# In[28]:


# Calculate statistics for each regime
statistics = pd.DataFrame(index=regime_colors.keys())

# Mean Daily Return
statistics['Mean Daily Return'] = data.groupby('Regime Classification')['Returns'].mean()

# Daily Standard Deviation
statistics['Daily Standard Deviation'] = data.groupby('Regime Classification')['Returns'].std()

# Number of Days
statistics['Number of Days'] = data['Regime Classification'].value_counts()

# Mean Regime Length
statistics['Mean Regime Length'] = data['Regime Classification'].value_counts() / len(data['Regime Classification'].unique())

print(statistics)


# In[ ]:




