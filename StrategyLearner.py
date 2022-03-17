"""  		   	  			  	 		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
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
  		   	  			  	 		  		  		    	 		 		   		 		  
import datetime as dt  		   	  			  	 		  		  		    	 		 		   		 		  
import pandas as pd  		   	  			  	 		  		  		    	 		 		   		 		  
import util as ut  		   	  			  	 		  		  		    	 		 		   		 		  
import random  		 
import QLearner
import marketsimcode
import indicators 	  
import numpy as np	
import time		  	 		  	
import matplotlib.pyplot as plt 		
import ManualStrategy    	 	
import experiment1
import experiment2	 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
class StrategyLearner(object):  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    # constructor  		   	  			  	 		  		  		    	 		 		   		 		  
    def __init__(self, verbose = False, impact=0.0):  		   	  			  	 		  		  		    	 		 		   		 		  
        self.verbose = verbose  		   	  			  	 		  		  		    	 		 		   		 		  
        self.impact = impact  	
        self.small_reward = 1.0
        self.epsilon = 0.5
        self.episodes = 10
        self.n = 2

    def author(self):
        return 'atorreshotrum3'	   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    # this method should create a QLearner, and train it for trading  		   	  			  	 		  		  		    	 		 		   		 		  
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,12,31), \
        sv = 10000):  		   	  			  	 		  		  		    	 		 		   		 		  
        
        # Get symbol prices
        date_range = pd.date_range(start=sd, end=ed)	 
        prices_all = ut.get_data([symbol], date_range)  # automatically adds SPY  
        del prices_all['SPY']
       
        # Add the daily returns
        values = prices_all[symbol]
        daily_returns = values.copy()
        daily_returns[1:] = (values[1:] / values[:-1].values) - 1
        daily_returns.iloc[0] = 0 
        prices_all['Daily Returns'] = daily_returns

        # Compute the value of the Benchmark Portfolio
        benchmark_orders = pd.DataFrame({'Symbol': symbol, 'Order':['BUY', 'SELL'], 'Shares':1000}, index = [prices_all.index[0], prices_all.index[-1]])      
        benchmark = marketsimcode.compute_portvals(benchmark_orders, start_val = sv, commission=0.0, impact=self.impact)
        
        # Compute the cumulative returns of benchmark
        portvals = benchmark['Value']
        benchmark_cum_ret = (portvals[-1]/ portvals[0]) - 1 
        prev_ret = benchmark_cum_ret   

        # Get indicator data
        bbp = indicators.b_band(symbol, date_range)
        rsi = indicators.RSI(symbol, date_range)
        macd = indicators.MACD(symbol, date_range)

        # Put indicator data in an indicator DF
        data = {'BBP':bbp, 'RSI':rsi, 'MACD':macd}
        self.indicator_df = pd.DataFrame(data, index = prices_all.index)
                
        # Instantiate Q Learner - 3 actions (BUY, SELL, HOLD), 81 States
        #                        (3 for each indicator, 3 for each position, 3^4 = 81)
        self.ql = QLearner.QLearner(num_states=81, num_actions=3, alpha = 0.2, gamma = 0.9, \
                                rar = self.epsilon, radr = 0.99, dyna = 0)
        
        # Initialize convergence variables
        converged = False
        episodes = 1
        prev_ret = 0.0
        second_run = False

        while not converged:

            # Make DF Trades filled with zeros
            df_trades =  pd.DataFrame(0.0, index = prices_all.index, columns=['Trades'])

            # Start off holding 0 stocks
            current_holdings = 0

            # Initial state --> 0 = hold indication, 1 = short indication, 2 = long indication
            s = self.to_state(current_holdings, time_step=0)
            a = self.ql.querysetstate(s)

            # Execute initial action
            if a == 2:
                df_trades ['Trades'].iloc[0] = 1000.0
            elif a == 1:
                df_trades ['Trades'].iloc[0] = -1000.0
            
            # Update current holdings
            current_holdings = df_trades ['Trades'].iloc[0]
            
            # Get reward 
            r = self.get_reward(prices_all, time_step=0, holdings = current_holdings)

            # Calculate new state
            s_prime = self.to_state(current_holdings, time_step=1)

            # For the rest of the days in the trading period
            for i in range(len(df_trades[1:])):

                # Update Q matrix, get next action
                a = self.ql.query(s_prime, r)

                # Execute action, only if it is legal
                if a == 2 and current_holdings <= 0.0:
                    df_trades ['Trades'].iloc[i+1] = 1000.0 - current_holdings
                    current_holdings = 1000
                elif a == 1 and current_holdings >= 0.0:
                    df_trades ['Trades'].iloc[i+1] = -1000.0 - current_holdings
                    current_holdings = -1000

                # Get reward
                r = self.get_reward(prices_all, time_step=i+1, holdings = current_holdings)

                # Get new state
                s_prime = self.to_state(current_holdings, time_step=i+1)
            
            
            # Calculate the cum_ret for the rest of the trading days
            orders_df = pd.DataFrame(index = prices_all.index, columns=['Symbol', 'Order', 'Shares'])
            orders_df['Symbol'] = symbol
            
            orders_df['Shares'] = df_trades['Trades']

            # Set Buy and Sell
            orders_df.loc[orders_df['Shares'] < 0, ['Order']] = 'SELL'
            orders_df.loc[orders_df['Shares'] > 0, ['Order']] = 'BUY'

            #Drop NaNs
            orders_df.dropna(inplace=True)

            # Make everything positive
            orders_df['Shares'] = orders_df['Shares'].abs()

            # If orders sheet is not empty
            if len(orders_df) != 0:
            # Compute the value of the Manual Portfolio
                learner_strategy = marketsimcode.compute_portvals(orders_df, start_val = sv, commission=0.0, impact=self.impact)
                
                # Compute the cum_ret of the learner
                learner_portvals = learner_strategy['Value']
                learner_cum_ret = (learner_portvals[-1]/ learner_portvals[0]) - 1 
            else:
                learner_cum_ret = 0.0

            # Update convergence criteria
            if learner_cum_ret != 0.0 and (learner_cum_ret - prev_ret) < 0.001 and episodes > 5:
                if learner_cum_ret > 1.5 or second_run:
                    converged = True
                else:
                    self.ql = QLearner.QLearner(num_states=81, num_actions=3, alpha = 0.2, gamma = 0.9, \
                                rar = self.epsilon, radr = 0.99, dyna = 0)
                    episodes = 0
                    prev_ret = 0.0
                    second_run = True 
            elif episodes > 20:
                if learner_cum_ret > benchmark_cum_ret or second_run:
                    converged = True
                else:
                    self.ql = QLearner.QLearner(num_states=81, num_actions=3, alpha = 0.2, gamma = 0.9, \
                                rar = self.epsilon, radr = 0.99, dyna = 0)
                    episodes = 0
                    prev_ret = 0.0
                    second_run = True 
            elif learner_cum_ret == 0.0:
                self.ql = QLearner.QLearner(num_states=81, num_actions=3, alpha = 0.2, gamma = 0.9, \
                                rar = self.epsilon, radr = 0.99, dyna = 0)
                episodes += 1
                prev_ret = 0.0
            else:
                episodes += 1
                prev_ret = learner_cum_ret
                

        #print(benchmark_cum_ret)
        #print(learner_cum_ret)        
        #return df_trades
        #return learner_cum_ret, benchmark_cum_ret



    # this method should test the policy  		   	  			  	 		  		  		    	 		 		   		 		  
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2010,1,1), \
        ed=dt.datetime(2011,12,31), \
        sv = 10000):  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
        # Get symbol prices
        date_range = pd.date_range(start=sd, end=ed)	 
        prices_all = ut.get_data([symbol], date_range)  # automatically adds SPY  
        del prices_all['SPY']
        
        # Add the daily returns
        values = prices_all[symbol]
        daily_returns = values.copy()
        daily_returns[1:] = (values[1:] / values[:-1].values) - 1
        daily_returns.iloc[0] = 0 
        prices_all['Daily Returns'] = daily_returns

        # Compute the value of the Benchmark Portfolio
        benchmark_orders = pd.DataFrame({'Symbol': symbol, 'Order':['BUY', 'SELL'], 'Shares':1000}, index = [prices_all.index[0], prices_all.index[-1]])
        benchmark = marketsimcode.compute_portvals(benchmark_orders, start_val = sv, commission=0.0, impact=self.impact)
        
        # Compute the cumulative returns of benchmark
        portvals = benchmark['Value']
        benchmark_cum_ret = (portvals[-1]/ portvals[0]) - 1 

        # Get indicator data
        bbp = indicators.b_band(symbol, date_range)
        rsi = indicators.RSI(symbol, date_range)
        macd = indicators.MACD(symbol, date_range)

        # Put indicator data in an indicator DF
        data = {'BBP':bbp, 'RSI':rsi, 'MACD':macd}
        self.indicator_df = pd.DataFrame(data, index = prices_all.index)

        # Make DF Trades filled with zeros
        df_trades =  pd.DataFrame(0.0, index = prices_all.index, columns=['Trades'])

        # Start off holding 0 stocks
        current_holdings = 0

        # Initial state --> 0 = hold indication, 1 = short indication, 2 = long indication
        s = self.to_state(current_holdings, time_step=0)
        a = self.ql.querysetstate(s)

        # Execute initial action
        if a == 2:
            df_trades ['Trades'].iloc[0] = 1000.0
            current_holdings = 1000
        elif a == 1:
            df_trades ['Trades'].iloc[0] = -1000.0
            current_holdings = -1000 

        s = self.to_state(current_holdings, time_step=1)


        for i in range(len(df_trades[1:])):

            # Get state and action state --> 0 = hold indication, 1 = short indication, 2 = long indication
            a = self.ql.querysetstate(s)

            # Execute action, only if it is legal
            if a == 2 and current_holdings <= 0.0:
                df_trades ['Trades'].iloc[i+1] = 1000.0 - current_holdings
                current_holdings = 1000
            elif a == 1 and current_holdings >= 0.0:
                df_trades ['Trades'].iloc[i+1] = -1000.0 - current_holdings
                current_holdings = -1000 

            s = self.to_state(current_holdings, time_step=i+1)
        
        
        # Calculate the cum_ret for the rest of the trading days
        orders_df = pd.DataFrame(index = prices_all.index, columns=['Symbol', 'Order', 'Shares'])
        orders_df['Symbol'] = symbol
        
        orders_df['Shares'] = df_trades['Trades']

        # Set Buy and Sell
        orders_df.loc[orders_df['Shares'] < 0, ['Order']] = 'SELL'
        orders_df.loc[orders_df['Shares'] > 0, ['Order']] = 'BUY'

        #Drop NaNs
        orders_df.dropna(inplace=True)

        # Make everything positive
        orders_df['Shares'] = orders_df['Shares'].abs()

        # If orders sheet is not empty
        if len(orders_df) != 0:
        # Compute the value of the Manual Portfolio
            learner_strategy = marketsimcode.compute_portvals(orders_df, start_val = sv, commission=0.0, impact=self.impact)
            
            # Compute the cum_ret of the learner
            learner_portvals = learner_strategy['Value']
            learner_cum_ret = (learner_portvals[-1]/ learner_portvals[0]) - 1 
        else:
            learner_cum_ret = 0.0

        #print(benchmark_cum_ret)
        #print(learner_cum_ret)
        return df_trades#, learner_cum_ret, benchmark_cum_ret
        #return df_trades



    # Calculate the reward
    def get_reward(self, prices_all, time_step, holdings):
        if holdings < 0:
            position = -1
        elif holdings > 0:
            position = 1
        else:
            position = 0
        if (time_step+1) < len(prices_all):
            return position * prices_all['Daily Returns'].iloc[time_step+1]*(1-self.impact)
        else: 
            return position * prices_all['Daily Returns'].iloc[time_step]*(1+self.impact)

    # Get the inline state number   	  			  	 		  		  		    	 		 		   		 		  
    def to_state(self, current_holdings, time_step):

        if current_holdings < 0:
            position = 1
        elif current_holdings > 0:
            position = 2
        else:
            position = 0

        s = [self.get_bbp_state(self.indicator_df, time_step), \
            self.get_rsi_state(self.indicator_df, time_step), \
                self.get_macd_state(self.indicator_df, time_step), position]

        return int(s[0]*27+s[1]*9+s[2]*3+s[3])

    # Get the state of the BBP
    def get_bbp_state(self, indicator_df, index):
        value = indicator_df['BBP'].iloc[index]
        if value > 100:
            return 1
        elif value < 0:
            return 2
        else:
            return 0

    # Get the state of the RSI
    def get_rsi_state(self, indicator_df, index):
        value = indicator_df['RSI'].iloc[index]
        if value > 70:
            return 1
        elif value < 30:
            return 2
        else:
            return 0

    # Get the state of the MACD
    def get_macd_state(self, indicator_df, index):
        value = indicator_df['MACD'].iloc[index]
        # If the first day
        if index == 0:
            return 0
        else:
            prev_value = indicator_df['MACD'].iloc[index-1]
            if prev_value < 0 and value > 0:
                return 2
            elif prev_value > 0 and value < 0:
                return 1
            else:
                return 0

    # From df_trades, compute cumulative return
    def df_to_cumret(self, symbol, sd, ed, df_trades, sv):
        
        # Get symbol prices
        date_range = pd.date_range(start=sd, end=ed)	 
        prices_all = ut.get_data([symbol], date_range)  # automatically adds SPY  
        del prices_all['SPY']
        
        # Calculate the cum_ret for the rest of the trading days
        orders_df = pd.DataFrame(index = prices_all.index, columns=['Symbol', 'Order', 'Shares'])
        orders_df['Symbol'] = symbol
        
        orders_df['Shares'] = df_trades['Trades']

        # Set Buy and Sell
        orders_df.loc[orders_df['Shares'] < 0, ['Order']] = 'SELL'
        orders_df.loc[orders_df['Shares'] > 0, ['Order']] = 'BUY'

        #Drop NaNs
        orders_df.dropna(inplace=True)

        # Make everything positive
        orders_df['Shares'] = orders_df['Shares'].abs()

        # If orders sheet is not empty
        if len(orders_df) != 0:
        # Compute the value of the Manual Portfolio
            learner_strategy = marketsimcode.compute_portvals(orders_df, start_val = sv, commission=0.0, impact=self.impact)
            
            # Compute the cum_ret of the learner
            learner_portvals = learner_strategy['Value']
            learner_cum_ret = (learner_portvals[-1]/ learner_portvals[0]) - 1 
        else: 
            learner_portvals = pd.DataFrame(index = prices_all.index, columns=['Value'])
            learner_portvals['Value'] = sv
            learner_cum_ret = 0.0
        
        return learner_cum_ret, learner_portvals
  		   	  			  	 		  		  		    	 		 		   		 		  
if __name__=="__main__":  	

    experiment1.experiment1()
    experiment2.experiment2()

    """
    symbol = 'ML4T-220' 	   
    sv = 100000
    in_sample = False
    impact  = 0
    #random.seed(30)
    beginning = time.time()

    if in_sample:
        SD=dt.datetime(2008, 1, 1)
        ED=dt.datetime(2009,12,31)
    else:
        SD=dt.datetime(2010, 1, 1)
        ED=dt.datetime(2011,12,31)

    
    ###################
    # TESTING SEVERAL TIMES
    loops = 50
    ind = np.arange(loops)
    returns = pd.DataFrame(index = ind, columns = ['IS_DR', 'IS_BM', 'OS_DR', 'OS_BM'])
    start_time = time.time()
    time_counter = 0


    for i in range(loops):
        start_time = time.time()
        sl = StrategyLearner()
        is_dr, is_bm = sl.addEvidence(symbol=symbol, sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = sv)
        if((time.time() - start_time) >25):
            time_counter += 1
        trades, os_dr, os_bm = sl.testPolicy(symbol=symbol, sd=SD, ed=ED, sv = sv)

        returns['IS_DR'].iloc[i] = is_dr
        returns['IS_BM'].iloc[i] = is_bm
        returns['OS_DR'].iloc[i] = os_dr
        returns['OS_BM'].iloc[i] = os_bm

    print(returns)
    alice = returns.where(returns['OS_DR']<1)#returns['OS_BM'])
    alice.dropna(inplace = True)
    print(symbol)
    print("Num Negative: ", len(alice))
    print("Num OT: ", time_counter)

    returns.plot()
    plt.axhline(y=2)
    plt.show()
    ####################################################################
    
    
    ####################################################
    # IMPACT TESTING, DELETE THIS
    #####################################################
    datapoints = 10
    impact  = 0
    increment = 0.0025
    index = []
    for i in range(datapoints):
        index.append(impact + increment * i)
    
    returns = pd.DataFrame(index = index, columns = ["Cumulative Returns", "Differences"])

    for i in range(datapoints):
        start_time = time.time()
        sl = StrategyLearner(impact=impact)  
        sl.addEvidence(symbol=symbol, sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = sv)
        trades, cr, br = sl.testPolicy(symbol=symbol, sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = sv)
        returns['Cumulative Returns'].iloc[i] = cr
        impact += increment
        if i > 0:
            diff = trades.values != original_trades.values
            result = diff.flatten().sum()
            returns['Differences'].iloc[i] = result
        else:
            returns['Differences'].iloc[i] = 0
            original_trades = trades


    print(returns)
    fig, ax = plt.subplots()
    ax.plot(returns['Cumulative Returns'])
    ax.set_xlabel("Impact")
    ax.set_ylabel("Cumulative Returns")
    plt.title('Figure 3: Experiment 2 Results')
    plt.savefig('Figure_3')
    plt.show()
    
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
    plt.show()
    
    plt.clf()
    plt.cla()
    plt.close()
    print("TIME: ", ((time.time() - beginning))/60)

    # REMEMBER TO DELETE THE CR RETURN OF testPolicy()!!!!!

    #######################################################################
    


    
    
    start_time = time.time()
    sl = StrategyLearner()

    print("ADD EVIDENCE") 
    sl.addEvidence(symbol=symbol, sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = sv)			  	 		  		  		    	 		 		   		 		  
    print("--- %s seconds ---" % (time.time() - start_time))
    print(" ")

    print("1st IN SAMPLE") 
    start_time = time.time()
    trades = sl.testPolicy(symbol=symbol, sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = sv)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(" ")

    print("2ND IN SAMPLE") 
    start_time = time.time()
    trades = sl.testPolicy(symbol=symbol, sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = sv)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(" ")

    print("1ST OUT SAMPLE")
    start_time = time.time()
    trades = sl.testPolicy(symbol=symbol, sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = sv)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(" ")

    print("SECOND OUT SAMPLE")
    start_time = time.time()
    trades = sl.testPolicy(symbol=symbol, sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = sv)
    print("--- %s seconds ---" % (time.time() - start_time))
    """
    