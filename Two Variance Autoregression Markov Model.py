#!/usr/bin/env python
# coding: utf-8

# In[8]:


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

# Drop NaN values
data = data.dropna()

# Compute daily returns
data['Returns'] = data['Close'].pct_change()

# Drop NaN values from returns
data = data.dropna()

# Fit the model
model = sm.tsa.MarkovAutoregression(data['Returns'], k_regimes=2, order=1, switching_variance=True)
res = model.fit(maxiter=1000)

# Combine them into one array
start_params = np.array(ar_params + variance_params + transition_params)

# Get smoothed probabilities
data['Smoothed Probability 0'] = res.smoothed_marginal_probabilities[0]
data['Smoothed Probability 1'] = res.smoothed_marginal_probabilities[1]

# Plot smoothed probabilities
fig, axs = plt.subplots(2, 1, figsize=(14, 12))

axs[0].plot(data.index, data['Smoothed Probability 0'], 'b-',
            label='Smoothed probability of regime 0 (Low Variance)')
axs[0].set(title='Smoothed Probability of Low Variance Regime',
           ylabel='Probability',
           xlabel='Date')
axs[0].legend()

axs[1].plot(data.index, data['Smoothed Probability 1'], 'g-',
            label='Smoothed probability of regime 1 (High Variance)')
axs[1].set(title='Smoothed Probability of High Variance Regime',
           ylabel='Probability',
           xlabel='Date')
axs[1].legend()

# Set x-axis tick labels to display years
years = pd.date_range(start='2000-01-01', end='2017-12-31', freq='Y').year
for ax in axs:
    ax.set_xticks([pd.to_datetime(str(year)) for year in years])
    ax.set_xticklabels(years)
    ax.grid(True, color='white')
    ax.set_facecolor('grey')

plt.tight_layout()
plt.show()

# Print the model parameters and summary
print(res.params)
print(res.summary())


# In[2]:


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

# Create a subplot
fig, ax1 = plt.subplots(figsize=(14, 6))

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

plt.tight_layout()
plt.show()

# Print the model parameters and summary
print(res.params)
print(res.summary())


# In[ ]:




