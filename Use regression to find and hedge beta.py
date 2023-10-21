#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from openbb_terminal.sdk import openbb
import statsmodels.api as sm
from statsmodels import regression 


# In[11]:


plt.style.use("default")
plt.rcParams["figure.figsize"] = [5.5, 4.0]
plt.rcParams["figure.dpi"] = 140
plt.rcParams["lines.linewidth"] = 0.75
plt.rcParams["font.size"] = 8


# In[20]:


symbols = ["NEM", "RGLD", "SSRM", "TSLA", "LLY", "UNH", "JNJ", "MRK", "SPY"]
data = openbb.economy.index(
    symbols, 
    start_date="2015-01-01", 
    end_date="2023-12-31"
)


# In[13]:


data


# In[21]:


#.pop removes the SPY column from the original data frame then creates its own data frame of the returns.
benchmark_returns = (
    data
    .pop("SPY")
    .pct_change()
    .dropna()
)


# In[22]:


#creates a new column in the data frame that is the sum of all the returns in the portfolio
portfolio_returns = (
    data
    .pop("TSLA")
    .pct_change()
    .dropna()
)


# In[23]:


portfolio_returns.plot()
benchmark_returns.plot()
plt.ylabel("Daily Returns")
plt.legend()


# In[24]:


X = benchmark_returns.values
Y = portfolio_returns.values

def linreg(x, y):
    #Add a column a 1s to fit falpha
    x = sm.add_constant(x)
    model = regression.linear_model.OLS(y, x).fit()
    #Remove the constant now that we're done
    x=x[:, 1]
    return model.params[0], model.params[1]

alpha, beta = linreg(X,Y)
print(f"alpha: {alpha}")
print(f"Beta: {beta}")


# In[25]:


X2 = np.linspace(X.min(), X.max(), 100)
Y_hat = X2 * beta + alpha

#Plot the raw data
plt.scatter(X, Y, alpha=0.3)
plt.xlabel("SPY daily return")
plt.ylabel("Portfolio daily return")

#Add the regression line 
plt.plot(X2, Y_hat, "r", alpha=0.9)


# In[17]:


#Construct a portfolio with beta hedging
hedged_portfolio_returns = -1 * beta * benchmark_returns + portfolio_returns


# In[18]:


P = hedged_portfolio_returns.values
alpha, beta = linreg(X, P)
print(f"Alpha:{alpha}")
print(f"Beta: {round(beta,6)}")


# In[20]:


def information_ratio(
    portfolio_returns, 
    benchmar_returns
):
    """
    Parameters
    ----------
    portfolio_returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
    benchmark_returns : int, float
        Daily returns of the benchmark or factor, noncumulative.
    Returns
    -------
    information_ratio : float
    Note
    -----
    See https://en.wikipedia.org/wiki/Information_ratio for more details.
    """
    
    active_return = portfolio_returns - benchmark_returns
    tracking_error = active_return.std()
    
    return active_return.mean() / tracking_error


# In[22]:


hedged_ir = information_ratio(
    hedged_portfolio_returns, 
    benchmark_returns
)
unhedged_ir=information_ratio(
    portfolio_returns,
    benchmark_returns
)
print(f"Hedged information ratio: {hedged_ir}")
print(f"Unhedged information ratio:{unhedged_ir}")


# In[ ]:




