#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pandas_datareader.data as web

import matplotlib.pyplot as plt

from zipline import run_algorithm
from zipline.api import (
    attach_pipeline,
    date_rules,
    time_rules,
    get_datetime,
    order_target_percent,
    pipeline_output,
    record,
    schedule_function,
    get_open_orders,
    calendars,
    set_commission,
    set_slippage,
)
from zipline.finance import commission, slippage
from zipline.pipeline import Pipeline, CustomFactor
from zipline.pipeline.factors import Returns, AverageDollarVolume

import pyfolio as pf

from alphalens.utils import get_clean_factor_and_forward_returns
from alphalens.tears import create_full_tear_sheet

import warnings

warnings.filterwarnings("ignore")


# In[2]:


# Settings
MONTH = 21
YEAR = 12 * MONTH
N_LONGS = 50
N_SHORTS = 50
VOL_SCREEN = 500 # in thousands


# In[3]:


class MeanReversion(CustomFactor):
    # Returns automatically captures the returns agains the
    # window length you pass
    inputs = [Returns(window_length=MONTH)]
    window_length = YEAR

    # The compute method is passed these arguments
    def compute(self, today, assets, out, monthly_returns):
        # Create a DataFrame of monthly returns, then substract
        # the latest return from the mean and divide by the std dev
        df = pd.DataFrame(monthly_returns)
        out[:] = df.iloc[-1].sub(df.mean()).div(df.std())


# In[5]:


def compute_factors():
    mean_reversion = MeanReversion()
    dollar_volume = AverageDollarVolume(window_length=30)
    pipe = Pipeline(
        columns={
            "longs": mean_reversion.bottom(N_LONGS),
            "shorts": mean_reversion.top(N_SHORTS),
            "ranking": mean_reversion.rank(ascending=False),
        },
        screen=dollar_volume.top(VOL_SCREEN),
    )
    return pipe


# In[6]:


mean_reversion = MeanReversion()
dollar_volume = AverageDollarVolume(window_length=30)
Pipeline(
    columns={
        "longs": mean_reversion.bottom(N_LONGS),
        "shorts": mean_reversion.top(N_SHORTS),
        "ranking": mean_reversion.rank(ascending=False),
    },
#     screen=dollar_volume.top(VOL_SCREEN),
).show_graph()


# In[7]:


def before_trading_start(context, data):
    context.factor_data = pipeline_output("factor_pipeline")
    record(factor_data=context.factor_data.ranking)

    assets = context.factor_data.index
    record(prices=data.current(assets, "price"))


# In[8]:


def rebalance(context, data):
    # Get the factor data and the assets we care about.
    factor_data = context.factor_data
    assets = factor_data.index

    # Filter the assets we want to go long, short, and divest
    longs = assets[factor_data.longs]
    shorts = assets[factor_data.shorts]
    divest = context.portfolio.positions.keys() - longs.union(shorts)

    # Print some portfolio details.
    print(
        f"{get_datetime().date()} | Longs {len(longs)} | Shorts | {len(shorts)} | {context.portfolio.portfolio_value}"
    )

    # Execute the trades
    exec_trades(data, assets=divest, target_percent=0)
    exec_trades(data, assets=longs, target_percent=1 / N_LONGS if N_LONGS else 0)
    exec_trades(data, assets=shorts, target_percent=-1 / N_SHORTS if N_SHORTS else 0)


# In[9]:


def exec_trades(data, assets, target_percent):
    # Loop through every asset...
    for asset in assets:
        # ...if the asset is tradeable and there are no open orders...
        if data.can_trade(asset) and not get_open_orders(asset):
            # ...execute the order against the target percent
            order_target_percent(asset, target_percent)


# In[10]:


def initialize(context):
    # Initialize the algorithm by attaching the factor pipeline and
    # setting up the scheduling function to run weekly, at market open,
    # using the US equities calendar.
    attach_pipeline(compute_factors(), "factor_pipeline")
    schedule_function(
        rebalance,
        date_rules.week_start(),
        time_rules.market_open(),
        calendar=calendars.US_EQUITIES,
    )

    # Set up the commission model to charge us per share and a volume slippage model
    set_commission(us_equities=commission.PerShare(cost=0.00075, min_trade_cost=0.01))
    set_slippage(
        us_equities=slippage.VolumeShareSlippage(volume_limit=0.0025, price_impact=0.01)
    )


# In[11]:


def analyze(context, perf):
    # Simple plot of the portfolio value
    perf.portfolio_value.plot()


# In[12]:


start = pd.Timestamp('2013-01-01')
end = pd.Timestamp('2014-01-01')
capital_base = 1e7


# In[13]:


sp500 = web.DataReader('SP500', 'fred', start, end).SP500
benchmark_returns = sp500.pct_change()


# In[14]:


perf = run_algorithm(
    start=start,
    end=end,
    initialize=initialize,
    analyze=analyze,
    capital_base=capital_base,
    benchmark_returns=benchmark_returns,
    before_trading_start=before_trading_start,
    bundle="quandl",
)


# In[15]:


perf.to_pickle("mean-reversion.pickle")


# In[16]:


perf = pd.read_pickle("mean-reversion.pickle")


# In[17]:


returns, positions, transactions = \
    pf.utils.extract_rets_pos_txn_from_zipline(perf)


# In[19]:


pf.create_full_tear_sheet(
    returns, positions=positions, transactions=transactions, round_trips=True,)


# In[20]:


prices = pd.concat(
    [df.to_frame(d) for d, df in perf.prices.dropna().items()], 
    axis=1
).T
prices.columns = [col.symbol for col in prices.columns]
prices.index = prices.index.normalize()


# In[21]:


factor_data = pd.concat(
    [df.to_frame(d) for d, df in perf.factor_data.dropna().items()], axis=1
).T
factor_data.columns = [col.symbol for col in factor_data.columns]
factor_data.index = factor_data.index.normalize()
factor_data = factor_data.stack()
factor_data.index.names = ["date", "asset"]


# In[22]:


alphalens_data = get_clean_factor_and_forward_returns(
    factor=factor_data, 
    prices=prices, 
    periods=(5, 10, 21, 63), 
    quantiles=5
)


# In[23]:


alphalens_data


# In[24]:


create_full_tear_sheet(alphalens_data)


# In[ ]:




