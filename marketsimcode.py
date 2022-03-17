"""MC2-P1: Market simulator.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		   	  			  	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		   	  			  	 		  		  		    	 		 		   		 		  
All Rights Reserved  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		   	  			  	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		   	  			  	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		   	  			  	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		   	  			  	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		   	  			  	 		  		  		    	 		 		   		 		  
or edited.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		   	  			  	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		   	  			  	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		   	  			  	 		  		  		    	 		 		   		 		  
GT honor code violation.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Student Name: Alonso Torres-Hotrum	   	  			  	 		  		  		    	 		 		   		 		  
GT User ID: atorreshotrum3 		   	  			  	 		  		  		    	 		 		   		 		  
GT ID: 903475423			  	 		  		  		    	 		 		   		 		  
"""  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
import pandas as pd  		   	  			  	 		  		  		    	 		 		   		 		  
import numpy as np  		   	  			  	 		  		  		    	 		 		   		 		  
import datetime as dt  		   	  			  	 		  		  		    	 		 		   		 		  
import os  		   	  			  	 		  		  		    	 		 		   		 		  
from util import get_data, plot_data  	
import math	   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
def compute_portvals(orders, start_val = 1000000, commission=0.0, impact=0.00):  		   	  			  	 		  		  		    	 		 		   		 		  
    # this is the function the autograder will call to test your code  		   	  			  	 		  		  		    	 		 		   		 		  
    # NOTE: orders_file may be a string, or it may be a file object. Your  		   	  			  	 		  		  		    	 		 		   		 		  
    # code should work correctly with either input  		   	  			  	 		  		  		    	 		 		   		 		  
    # TODO: Your code here  		  

    # Sort on dates  	
    orders = orders.sort_index(kind = 'mergesort')	
    # Find all symbols
    syms = orders.Symbol.unique()
    syms = list(syms)

    # Build df of prices
    sd = orders.index[0]
    ed = orders.index[-1]
    dates = pd.date_range(sd, ed) 	  	 		  		  		    	 		 		   		 		  
    prices_all = get_data(syms, dates)  # automatically adds SPY  		   	  			  	 		  		  		    	 		 		   		 		  
    prices = prices_all[syms]  # only portfolio symbols  
    SPY_prices = prices_all['SPY']	
    dates_final = SPY_prices.index

    # add cash column
    length =  len(prices)
    cash = np.ones(length)
    prices['Cash'] = cash

    # Make trades df
    trade_cols = syms
    trade_cols.append('Cash')
    trade_index = orders.index.unique()
    trades = pd.DataFrame(index= trade_index, columns = trade_cols)
    trades.fillna(0.0, inplace=True) 

    # Populate trades df
    for index, row in orders.iterrows():
        symb = row['Symbol']
        if row['Order'] == 'BUY':
            trades.loc[index][symb] += row['Shares']
            value = prices.loc[index][symb]
            trades.loc[index]['Cash'] -= ((1+impact)*(value *row['Shares']) + commission)
        else:
            trades.loc[index][symb] -= row['Shares']
            value = prices.loc[index][symb]
            trades.loc[index]['Cash'] +=  ((1-impact)*(value *row['Shares']) - commission)
    

    #make df holdings
    holdings = pd.DataFrame(index= dates_final, columns = trade_cols)
    holdings.fillna(0.0, inplace=True) 
        
    #populate holdings
    first_go = True
    for index, row in holdings.iterrows():
        if not first_go:
            holdings.loc[index] = holdings.loc[index] + prev_row
            if index in trades.index:
                holdings.loc[index] = (holdings.loc[index] + trades.loc[index])
            else:
                holdings.loc[index] = prev_row
            prev_row = holdings.loc[index]
        else:
            if index in trades.index:
                row = (row+trades.loc[index])
            row['Cash'] += start_val
            holdings.loc[index] = row
            prev_row = row
            first_go = False
    

    #make DF value
    value = holdings * prices
    value.dropna(inplace=True)

    # make portfolio value
    port_val = pd.DataFrame(index= dates_final, columns = ['Value'])
    port_val['Value'] = value.sum(axis = 1)
	  	 		  		  		    	 		 		   		 		  
    return port_val  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
def test_code(orders, start_val = 1000000, commission=0.0, impact=0.00):  		   	  			  	 		  		  		    	 		 		   		 		  
    # this is a helper function you can use to test your code  		   	  			  	 		  		  		    	 		 		   		 		  
    # note that during autograding his function will not be called.  		   	  			  	 		  		  		    	 		 		   		 		  
    # Define input parameters  		   	  			  	 		  		  		    	 		 		   		 		  
  	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    # Process orders  		   	  			  	 		  		  		    	 		 		   		 		  
    portvals = compute_portvals(orders, start_val)  		   	  			  	 		  		  		    	 		 		   		 		  
    if isinstance(portvals, pd.DataFrame):  		   	  			  	 		  		  		    	 		 		   		 		  
        portvals = portvals[portvals.columns[0]] # just get the first column  	   	  			  	 		  		  		    	 		 		   		 		  
    else:  		   	  			  	 		  		  		    	 		 		   		 		  
        "warning, code did not return a DataFrame"  		   	  			  	 		  		  		    	 		 		   		 		  
     		  		    	 		 		   		 		  
    # Get portfolio stats  		   	  			  	 		  		  		    	 		 		   		 		  
    # Get SPY info
    sd = portvals.index[0]
    ed = portvals.index[-1]
    dates = pd.date_range(sd, ed)  			  	 		  		  		    	 		 		   		 		  
    prices_SPY = get_data(['$SPX'], dates)
    prices_SPY = prices_SPY[prices_SPY.columns[1]]

    # Calcs for portfolio
    cum_ret = (portvals[-1]/ portvals[0]) - 1
    daily_returns = portvals.copy()
    daily_returns[1:] = (portvals[1:] / portvals[:-1].values) - 1
    daily_returns.iloc[0] = 0 
    avg_daily_ret = daily_returns[1:].mean()
    std_daily_ret = daily_returns[1:].std()
    sharpe_ratio = math.sqrt(252.0) * (avg_daily_ret / std_daily_ret)   

    #Calcs for SPY 		   	  			  	 		  		  		    	 		 		   		 		  
    cum_ret_SPY = (prices_SPY[-1]/ prices_SPY[0]) - 1
    daily_returns = prices_SPY.copy()
    daily_returns[1:] = (prices_SPY[1:] / prices_SPY[:-1].values) - 1
    daily_returns.iloc[0] = 0    
    avg_daily_ret_SPY = daily_returns[1:].mean()
    std_daily_ret_SPY = daily_returns[1:].std()
    sharpe_ratio_SPY = math.sqrt(252.0) * (avg_daily_ret_SPY / std_daily_ret_SPY)



    # Compare portfolio against $SPX  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Date Range: {sd} to {ed}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of Fund: {cum_ret}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Standard Deviation of Fund: {std_daily_ret}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Average Daily Return of Fund: {avg_daily_ret}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")  		   	  			  	 		  		  		    	 		 		   		 		  
    print()  		   	  			  	 		  		  		    	 		 		   		 		  
    print(f"Final Portfolio Value: {portvals[-1]}")  		   	  			  	 		  		  		    	 		 		   		 		  


def author():  		   	  			  	 		  		  		    	 		 		   		 		  
    return 'atorreshotrum3'   		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  


if __name__ == "__main__":  
    orders = pd.read_csv("./orders/orders-01.csv", index_col='Date', parse_dates=True)
   	  			  	 		  		  		    	 		 		   		 		  
    test_code(orders, start_val = 1000000, commission=0.00, impact=0.00)
 		   	  			  	 		  		  		    	 		 		   		 		  
