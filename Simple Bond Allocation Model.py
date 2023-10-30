#!/usr/bin/env python
# coding: utf-8

# In[146]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import quantstats as qs
from openbb_terminal.sdk import openbb


# In[147]:


symbols = ["^TYX", "^IRX", "^DJU", "TLT"]
start_date = "1990-01-01"
end_date = "2023-10-20"


# In[148]:


stocks = []
for symbol in symbols:
    df = (openbb.stocks.load(symbol, start_date = start_date, end_date = end_date, verbose=False)
         .drop(["Close", "Dividends", "Stock Splits"], axis=1)
         )
    df["symbol"] = symbol
    stocks.append(df)
    
prices = pd.concat(stocks)
prices.columns = ["open", "high", "low", "close", "volume", "symbol"]


# In[149]:


prices


# In[150]:


prices = prices_pivot = prices.pivot(columns='symbol')


# In[151]:


prices.rename(columns={"^DJU": "util", "^TYX": "30y", "^IRX": "tbill", "TLT": "tlt"}, inplace=True)


# In[152]:


prices["tlt_return"] = prices["close"]["tlt"].pct_change()


# In[153]:


prices["30y_ma"] = prices["close"]["30y"].rolling(window=10*5).mean()
prices["tbill_ma"] = prices["close"]["tbill"].rolling(window=38*5).mean()
prices["dju_ma"] = prices["close"]["util"].rolling(window=10*5).mean()


# In[154]:


prices


# In[ ]:





# In[155]:


prices = prices.dropna()


# In[156]:


prices["condition_a"] = prices["close"]["30y"] < prices["30y_ma"]
prices["condition_b"] = prices["close"]["tbill"] < prices["tbill_ma"]
prices["condition_c"] = prices["close"]["util"] > prices["dju_ma"]


# In[157]:


prices["exit_a"] = prices["close"]["30y"] > prices["30y_ma"]
prices["exit_b"] = prices["close"]["tbill"] > prices["tbill_ma"]
prices["exit_c"] = prices["close"]["util"] < prices["dju_ma"]


# In[158]:


conditions = pd.DataFrame({"condition_a": prices["condition_a"], "condition_b": prices["condition_b"], "condition_c": prices["condition_c"],
                          "exit_a": prices["exit_a"], "exit_b": prices["exit_b"], "exit_c": prices["exit_c"], 
                           "returns": prices["tlt_return"], "tlt_close": prices["close"]["tlt"]})


# In[159]:


buy_condition1 = conditions['condition_a'] & conditions['condition_b']
buy_condition2 = conditions['condition_a'] & conditions['condition_c']
buy_condition3 = conditions['condition_a'] & conditions['condition_b'] & conditions['condition_c']


# In[160]:


conditions['buy'] = buy_condition1 | buy_condition2 | buy_condition3


# In[161]:


sell_condition1 = conditions['exit_a'] & conditions['exit_b']
sell_condition2 = conditions['exit_a'] & conditions['exit_c']
sell_condition3 = conditions['exit_a'] & conditions['exit_b'] & conditions['exit_c']

conditions['sell'] = sell_condition1 | sell_condition2 | sell_condition3


# In[162]:


import quantstats as qs


# In[163]:


def apply_trading_strategy(data):
    initial_cash = 100000
    cash = initial_cash
    portfolio = {}
    portfolio_value = [initial_cash]
    


    for i, row in data.iterrows():
        # Exiting a position
        if 'asset' in portfolio and row['sell']:
            cash += portfolio['asset'] * row['tlt_close']
            del portfolio['asset']

        # Going long
        if row['buy'] and 'asset' not in portfolio:
            amount_to_buy = cash // row['tlt_close']
            portfolio['asset'] = amount_to_buy
            cash -= amount_to_buy * row['tlt_close']

        # Update the portfolio value
        total_value = cash + (portfolio.get('asset', 0) * row['tlt_close'])
        portfolio_value.append(total_value)
        
    portfolio_value = portfolio_value[1:]
    portfolio_value = pd.Series(portfolio_value, index=data.index)

    print(portfolio_value.head())
    qs.reports.full(portfolio_value)


# In[164]:


apply_trading_strategy(conditions)


# In[ ]:




