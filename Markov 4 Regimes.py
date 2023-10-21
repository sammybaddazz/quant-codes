#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import quantstats as qs
from openbb_terminal.sdk import openbb

# Constants
TICKER_SYMBOL = '^GSPC'
START_DATE = '2000-01-01'
END_DATE = '2017-12-31'
WINDOW_SIZE = 250
EMA_WINDOW = 20
ATR_WINDOW = 14
MULTIPLIER = 2

def fetch_data(ticker_symbol, start_date, end_date):
    return openbb.stocks.load(ticker_symbol, start_date=start_date, end_date=end_date)

def preprocess_data(data):
    data['Returns'] = data['Close'].pct_change()
    data.dropna(inplace=True)
    return data

def fit_markov_model(data):
    model = sm.tsa.MarkovAutoregression(data['Returns'], k_regimes=2, order=1, switching_variance=True)
    res = model.fit()
    data['Smoothed Probability 0'] = res.smoothed_marginal_probabilities[0]
    data['Smoothed Probability 1'] = res.smoothed_marginal_probabilities[1]
    return data

def triangular_moving_average(data, window_size):
    # First Simple Moving Average (SMA)
    sma = data['Close'].rolling(window=window_size).mean()
    
    # Second SMA on the first one to get TMA
    tma = sma.rolling(window=window_size).mean()
    
    return tma

def compute_atr(data, window):
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()     
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=window).mean()

def classify_regime(row):
    is_bullish = row['Close'] > row['TMA']
    is_bearish = not is_bullish
    high_var = row['Smoothed Probability 1'] >= 0.5

    if is_bullish and high_var:
        return 'Bullish with high variance'
    elif is_bullish:
        return 'Bullish with low variance'
    elif is_bearish and high_var:
        return 'Bearish with high variance'
    elif is_bearish:
        return 'Bearish with low variance'
    
def plot_results(data):
    plt.figure(figsize=(20, 14))
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

def apply_trading_strategy(data):
    initial_cash = 100000
    cash = initial_cash
    portfolio = {}
    portfolio_value = [initial_cash]

    for i, row in data.iterrows():
        # If we've bought an asset and the price crosses below the Keltner lower channel, sell it
        if 'asset' in portfolio and data['Close'].loc[i] < row['Keltner_Lower']:
            cash += portfolio['asset'] * row['Close']
            del portfolio['asset']

        # If we've shorted an asset and the price crosses above the Keltner upper channel, buy it back (cover the short)
        if 'shorted_asset' in portfolio and data['Close'].loc[i] > row['Keltner_Upper']:
            cash -= portfolio['shorted_asset'] * row['Close']  # use cash to cover the short
            del portfolio['shorted_asset']

        # Going long
        if row['Regime Classification'] == 'Bullish with low variance' and 'asset' not in portfolio:
            amount_to_buy = cash // row['Close']
            portfolio['asset'] = amount_to_buy
            cash -= amount_to_buy * row['Close']

        # Going short
        if row['Regime Classification'] == 'Bearish with low variance' and 'shorted_asset' not in portfolio:
            amount_to_short = cash // row['Close']
            portfolio['shorted_asset'] = amount_to_short
            portfolio['short_price'] = row['Close']
            cash += amount_to_short * row['Close']  # add cash from the proceeds of the short sale


        # Update the portfolio value
        total_value = cash + (portfolio.get('asset', 0) * row['Close']) - (portfolio.get('shorted_asset', 0) * (row['Close'] - 2*portfolio.get('short_price', 0)))
        portfolio_value.append(total_value)
        
    portfolio_value = portfolio_value[1:]
    portfolio_value = pd.Series(portfolio_value, index=data.index)

    long_exit = data[data['Regime Classification'] == 'Bullish with low variance']['Keltner_Lower']
    

    short_exit = data[data['Regime Classification'] == 'Bearish with low variance']['Keltner_Upper']
    
    print(portfolio_value.head())
    print(portfolio_value.index.dtype)
    qs.reports.full(portfolio_value)

def main():
    data = fetch_data(TICKER_SYMBOL, START_DATE, END_DATE)
    data = preprocess_data(data)
    data = fit_markov_model(data)
    data['TMA'] = triangular_moving_average(data, WINDOW_SIZE)
    data.dropna(inplace=True)
    data['EMA'] = data['Close'].ewm(span=EMA_WINDOW, adjust=False).mean()
    data['ATR'] = compute_atr(data, ATR_WINDOW)
    data['Keltner_Upper'] = data['EMA'] + data['ATR'] * MULTIPLIER
    data['Keltner_Lower'] = data['EMA'] - data['ATR'] * MULTIPLIER
    data['Regime Classification'] = data.apply(classify_regime, axis=1)
    apply_trading_strategy(data)
    plot_results(data)

if __name__ == '__main__':
    main()


# In[ ]:




