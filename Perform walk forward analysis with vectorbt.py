#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.stats as stats 
import vectorbt as vbt


# In[2]:


start = "2016-01-01 UTC"
end = "2020-01-01 UTC"
prices = vbt.YFData.download(
    "AAPL",
    start=start,
    end=end
).get("Close")


# In[3]:


(in_price, in_indexes), (out_price, out_indexes) = prices.vbt.rolling_split(
    n=30,
    window_len=365 * 2,
    set_lens=(180,),
    left_to_right=False,
)


# In[4]:


def simulate_all_params(price, windows, **kwargs):
    fast_ma, slow_ma = vbt.MA.run_combs(
        price, 
        windows, 
        r=2, 
        short_names=["fast", "slow"]
    )
    
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    
    pf = vbt.Portfolio.from_signals(price, entries, exits, **kwargs)
    
    return pf.sharpe_ratio()


# In[5]:


def get_best_index(performance, higher_better=True):
    if higher_better:
        return performance[
            performance.groupby("split_idx").idxmax()
        ].index
    
    return performance[
        performance.groupby("split_idx").idxmin()
    ].index


def get_best_params(best_index, level_name):
    return best_index.get_level_values(level_name).to_numpy()


# In[6]:


def simulate_best_params(price, best_fast_windows, best_slow_windows, **kwargs):
    fast_ma = vbt.MA.run(
        price, 
        window=best_fast_windows, 
        per_column=True
    )
    slow_ma = vbt.MA.run(
        price, 
        window=best_slow_windows, 
        per_column=True
    )
    
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    
    pf = vbt.Portfolio.from_signals(price, entries, exits, **kwargs)
    return pf.sharpe_ratio()


# In[7]:


windows = np.arange(10, 40)
in_sharpe = simulate_all_params(
    in_price, 
    windows, 
    direction="both",
    freq="d"
)


# In[8]:


in_best_index = get_best_index(in_sharpe)

in_best_fast_windows = get_best_params(
    in_best_index, 
    "fast_window"
)

in_best_slow_windows = get_best_params(
    in_best_index, 
    "slow_window"
)

in_best_window_pairs = np.array(
    list(
        zip(
            in_best_fast_windows, 
            in_best_slow_windows
        )
    )
)


# In[9]:


out_test_sharpe = simulate_best_params(
    out_price, 
    in_best_fast_windows, 
    in_best_slow_windows, 
    direction="both", 
    freq="d"
)


# In[10]:


out_test_sharpe


# In[11]:


in_sample_best = in_sharpe[in_best_index].values
out_sample_test = out_test_sharpe.values


# In[12]:


t, p = stats.ttest_ind(
    a=out_sample_test, 
    b=in_sample_best, 
    alternative="greater"
)
t,p


# In[13]:


out_test_sharpe


# In[19]:


in_sample_best = in_sharpe[in_best_index].values
out_sample_test = out_test_sharpe.values


# In[20]:


t, p = stats.ttest_ind(a=out_sample_test, b=in_sample_best, alternative="greater")
t,p


# In[21]:


out_test_sharpe


# In[ ]:




