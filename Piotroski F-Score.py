#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import yfinance as yf
import traceback

# Fetch financial data for a specific ticker and store it in global variables

balance_sheet = []
income_statement = []
cfs = []
years = []
profitability_score = 0
leverage_score = 0
operating_efficiency_score = 0
pe_ratio = 0 


# Read the list of S&P 500 companies
sp500_df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

# Get the list of tickers
tickers = sp500_df['Symbol'].tolist()


def get_data(ticker):
    global balance_sheet, income_statement, cfs, years, profitability_score
    ticker_obj = yf.Ticker(ticker)
    balance_sheet = ticker_obj.balance_sheet
    income_statement = ticker_obj.income_stmt
    cfs = ticker_obj.cash_flow
    years = balance_sheet.columns.intersection(income_statement.columns).intersection(cfs.columns)
    

def pe(ticker):
    global pe_ratio
    pe_ratio = yf.get_quote_table(ticker)['PE Ratio (TTM)']
    if pe_ratio != pe_ratio: #Check if NaN
        pe_ratio = 0
    
def profitability():
    #score #1 and 2 - net income
    global profitability_score 
    net_income = income_statement[years[0]]['Net Income']
    net_income_py = income_statement[years[1]]['Net Income']
    ni_score = 1 if net_income > 0 else 0
    ni_score_2 = 1 if net_income > net_income_py else 0 
    
    #score #3 - operating cash flow
    op_cf = cfs[years[0]]['Cash Flow From Continuing Operating Activities']
    op_cf_score = 1 if op_cf > 0 else 0
    
    #score #4 - change in RoA
    avg_assets = (balance_sheet[years[0]]['Total Assets']
                    + balance_sheet[years[1]]['Total Assets']) / 2
    avg_assets_py = (balance_sheet[years[1]]['Total Assets']
                    + balance_sheet[years[2]]['Total Assets']) / 2
    
    RoA = net_income / avg_assets 
    RoA_py = net_income_py / avg_assets_py 
    RoA_score = 1 if RoA > RoA_py else 0 
    
    #score #5 - Accurals
    
    total_assets = balance_sheet[years[0]]['Total Assets']
    accrual = op_cf / total_assets - RoA 
    ac_score = 1 if accrual > 0 else 0 
    
    profitability_score = ni_score + ni_score_2 + op_cf_score + RoA_score + ac_score
    print("Profitability score:" + str(profitability_score))
    
    
def leverage():
    global leverage_score
    #Score #6 - long-term debt ratio
    try:
        lt_debt = balance_sheet[years[0]]['Long Term Debt']
        total_assets = balance_sheet[years[0]]['Total Assets']
        debt_ratio = lt_debt / total_assets
        debt_ratio_score = 1 if debt_ratio < 0.4 else 0
    except: 
        debt_ratio_score = 1 
        
    #Score #7 - Current Ratio
    current_assets = balance_sheet[years[0]]['Total Assets']
    current_liab = balance_sheet[years[0]]['Total Liabilities Net Minority Interest']
    current_ratio = current_assets / current_liab
    current_ratio_score = 1 if current_ratio > 1 else 0 
    
    leverage_score = debt_ratio_score + current_ratio_score
    print("leverage score:" + str(leverage_score))
    
def operating_efficiency():
    global operating_efficiency_score
    #score #8- Gross Margin
    gp = income_statement[years[0]]['Gross Profit']
    gp_py = income_statement[years[1]]['Gross Profit']
    revenue = income_statement[years[0]]['Total Revenue']
    revenue_py = income_statement[years[1]]['Total Revenue']
    gm = gp/revenue
    gm_py = gp_py / revenue_py
    gm_score = 1 if gm > gm_py else 0 
    
    #score #9 - Asset turnover
    avg_assets = (balance_sheet[years[0]]['Total Assets']
                    + balance_sheet[years[1]]['Total Assets']) / 2
    avg_assets_py = (balance_sheet[years[1]]['Total Assets']
                    + balance_sheet[years[2]]['Total Assets']) / 2
                                  
    at = revenue / avg_assets
    at_py = revenue_py / avg_assets_py
    at_score = 1 if at > at_py else 0
    operating_efficiency_score = gm_score + at_score
    print("operating efficiency score: " + str(operating_efficiency_score))
    
     
        
for ticker in tickers [61:63]:
    try:
        get_data(ticker)
        print(ticker)
        profitability()
        leverage()
        operating_efficiency ()
    except Exception as e:
        print(f"{ticker} - Something went wrong: {str(e)}")
        traceback.print_exc() #print the full traceback


# In[ ]:




