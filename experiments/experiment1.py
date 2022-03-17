"""
Student Name: Alonso Torres-Hotrum		   	  			  	 		  		  		    	 		 		   		 		  
GT User ID: atorreshotrum3		   	  			  	 		  		  		    	 		 		   		 		  
GT ID: 903475423  		   	  			  	 		  		  		    	 		 		   		 		  
"""  		   	 

import datetime as dt  		   	  			  	 		  		  		    	 		 		   		 		  
import pandas as pd  		   	  			  	 		  		  		    	 		 		   		 		  
import util as ut  		   	  			  	 		  		  		    	 		 		   		 		  
import random  		 
import numpy as np
import marketsimcode	  	 		  	
import matplotlib.pyplot as plt 		
import ManualStrategy   
import StrategyLearner

def author():
    return 'atorreshotrum3'

def experiment1():  		   	 
    
    symbol = 'JPM' 	   
    sv = 100000
    in_sample = True
    impact  = 0
    #random.seed(30)

    if in_sample:
        SD=dt.datetime(2008, 1, 1)
        ED=dt.datetime(2009,12,31)
    else:
        SD=dt.datetime(2010, 1, 1)
        ED=dt.datetime(2011,12,31)

    # Set up learner
    sl = StrategyLearner.StrategyLearner()
    sl.addEvidence(symbol=symbol, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv = sv)
    sl_trades = sl.testPolicy(symbol=symbol, sd=SD, ed=ED, sv = sv)

    # Get cumulative return and portfolio value
    learner_cum_ret, learner_portvals = sl.df_to_cumret(symbol, SD, ED, sl_trades, sv=sv)
    
    # Set up manual learner
    ms = ManualStrategy.ManualStrategy()
    ms_df = ms.testPolicy(symbol, sd=SD, ed=ED, sv = 100000)

    # Get cumulative return and portfolio value
    manual_cum_ret, manual_portvals = sl.df_to_cumret(symbol, SD, ED, ms_df, sv = sv)
    
    # Get Benchmark return
    date_range = pd.date_range(start=SD, end=ED)	 
    prices_all = ut.get_data([symbol], date_range)  # automatically adds SPY  
    del prices_all['SPY']
    benchmark_orders = pd.DataFrame(index = [prices_all.index[0], prices_all.index[-1]], columns=['Symbol', 'Order', 'Shares'])
    benchmark_orders['Symbol'] = symbol
    benchmark_orders['Order'] = ['BUY', 'SELL']
    benchmark_orders['Shares'] = 1000
    bench_portvals = marketsimcode.compute_portvals(benchmark_orders, start_val = sv, commission=0.0, impact=impact)
    bench_temp = bench_portvals['Value']
    bench_cum_ret = (bench_temp[-1]/ bench_temp[0]) - 1 


    # Make Comparison DF
    date_range = pd.date_range(start=SD, end=ED)	 
    comparison = ut.get_data([symbol], date_range)  # automatically adds SPY  
    del comparison['SPY']
    comparison['Learner'] = learner_portvals
    comparison['Manual'] = manual_portvals
    comparison['Benchmark'] = bench_portvals
    comparison = comparison.ffill() 
    comparison = comparison.fillna(method='bfill')
    comparison = comparison / comparison.iloc[0]

    # Plot
    comparison.plot()
    plt.xlabel("In-Sample Dates")
    plt.ylabel("Normalized Portfolio Value")
    plt.title('Figure 2: Experiment 1 Normalized Comparison')
    plt.savefig('Figure_2')
    plt.clf()
    plt.cla()
    plt.close()

    
