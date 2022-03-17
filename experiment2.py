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
import time

def author():
    return 'atorreshotrum3'

def experiment2():  

    symbol = 'JPM' 	   
    sv = 100000

    SD=dt.datetime(2008, 1, 1)
    ED=dt.datetime(2009,12,31)

    # Set number of impact values to test, original impact and increment
    datapoints = 10
    impact  = 0
    increment = 0.0025
    index = []
    for i in range(datapoints):
        index.append(impact + increment * i)
    
    returns = pd.DataFrame(index = index, columns = ["Cumulative Returns", "Differences"])

    for i in range(datapoints):
        sl = StrategyLearner.StrategyLearner(impact=impact)  
        sl.addEvidence(symbol=symbol, sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = sv)
        trades = sl.testPolicy(symbol=symbol, sd=SD, ed=ED, sv = sv)
        cr, learner_portvals = sl.df_to_cumret(symbol, SD, ED, trades, sv=sv)
        
        returns['Cumulative Returns'].iloc[i] = cr
        impact += increment

    fig, ax = plt.subplots()
    ax.plot(returns['Cumulative Returns'])
    ax.set_xlabel("Impact")
    ax.set_ylabel("Cumulative Returns")
    plt.title('Figure 3: Experiment 2 Results')
    plt.savefig('Figure_3')
    plt.clf()
    plt.cla()
    plt.close()
    

    # Calculate the rolling mean and std
    r_mean = returns['Cumulative Returns'].rolling(window=3).mean()

    # Calculate STDev
    r_std = returns['Cumulative Returns'].rolling(window=3).std()	

    rolling = pd.DataFrame(index = index, columns = ["Rolling Mean", "Rolling STD"])
    rolling['Rolling Mean']  = r_mean
    rolling['Rolling STD'] = r_std

    # Plot
    rolling.plot()
    plt.xlabel("Impact")
    plt.ylabel("Value")
    plt.title('Figure 4: Experiment 2 Rolling Mean and Standard Deviation')
    plt.savefig('Figure_4')
    plt.clf()
    plt.cla()
    plt.close()
    