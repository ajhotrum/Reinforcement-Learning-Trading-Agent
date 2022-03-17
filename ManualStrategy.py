"""
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
import matplotlib.pyplot as plt
import marketsimcode
import indicators


class ManualStrategy:
    def __init__(self):
        pass

    def author(self):  		   	  			  	 		  		  		    	 		 		   		 		  
        return 'atorreshotrum3'

    def testPolicy(self, symbol = "AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000):
        
        # determine if in sample or out of sample
        OOS = True
        if sd==dt.datetime(2008, 1, 1):
            OOS = False
        
        # Benchmark
        in_sample_dates = pd.date_range(start=sd, end=ed)
        prices_all = get_data([symbol], in_sample_dates)  # automatically adds SPY  
        del prices_all['SPY']    

        # normalize
        symbol_prices = prices_all[symbol]       
        symbol_prices = symbol_prices/symbol_prices[0]
        prices_all[symbol] = symbol_prices

        # Calculate Daily Percent
        prices_all['Daily Percent'] = prices_all.pct_change(1)


        # Build DF for Buy/Sell orders
        # Make the orders DF and set all symbols to symbol
        orders_df = pd.DataFrame(index = prices_all.index, columns=['Symbol', 'Order', 'Shares'])
        orders_df['Symbol'] = symbol

        # Get BBP array
        bbp_df = indicators.b_band(symbol, in_sample_dates)
        bbp_array = np.array(bbp_df)
        # Set sell over 70, set buy under 30
        bbp_order = np.zeros(len(bbp_array))
        bbp_order[bbp_array > 100] = -1000
        bbp_order[bbp_array < 0] = 1000      


        # Get RSI array
        rsi_df = indicators.RSI(symbol, in_sample_dates)
        rsi_array = np.array(rsi_df)
        # Set sell over 70, set buy under 30
        rsi_order = np.zeros(len(rsi_array))
        rsi_order[rsi_array > 70] = -1000
        rsi_order[rsi_array < 30] = 1000


        # Get MCDP
        macd_dp = indicators.MACD(symbol, in_sample_dates)
        macd_array = np.array(macd_dp)
        # offset and multiply to see when sign changes
        macd_offset = np.insert(macd_array, 0, 0)
        macd_offset = np.delete(macd_offset, -1)
        macd_product =  np.multiply(macd_array, macd_offset)
        macd_product = np.where(macd_product < 0, 1000, 0)
        macd_sign = np.where(macd_array<0, -1, 1)
        macd_order = np.multiply(macd_product, macd_sign)


        # Add the resutling orders
        orders = np.add(macd_order, rsi_order)
        orders = np.add(orders, bbp_order)

        # Replace any 2000s and -2000s with 1000 and -1000
        orders[orders==2000] = 1000
        orders[orders==-2000] = -1000

        # Replace any 3000s and -3000s with 1000 and -1000
        orders[orders==3000] = 1000
        orders[orders==-3000] = -1000

        # Remove duplicates
        long_pos = False
        short = False
        for i in range(len(orders)):
            if orders[i] == 1000:
                if long_pos == True:
                    np.put(orders, i, 0.0)
                else:
                    long_pos = True
                    short = False
            elif orders[i] == -1000:
                if short == True:
                    np.put(orders, i, 0.0)
                else:
                    short = True
                    long_pos = False

        # Make it 2000 everywhere
        orders = orders * 2
        
        # Make the first order 1000
        for i in range(len(orders)):
            if orders[i] != 0.0:
                new_number = orders[i] * 0.5
                np.put(orders, i, new_number)
                break
                
        # Sell on last day
        for i in range(len(orders)):
            if orders[-i] != 0.0:
                new_number = orders[-i] * -0.5
                np.put(orders, -1, new_number)
                break

       
        # Make DF Trades
        df_trades =  pd.DataFrame(index = prices_all.index, columns=['Trades'])
        df_trades['Trades'] = orders

        # Build the orders sheet to pass to marketsim
        orders_df['Shares'] = orders

        # Set Buy and Sell
        orders_df.loc[orders_df['Shares'] < 0, ['Order']] = 'SELL'
        orders_df.loc[orders_df['Shares'] > 0, ['Order']] = 'BUY'

        #Drop NaNs
        orders_df.dropna(inplace=True)

        # Make everything positive
        orders_df['Shares'] = orders_df['Shares'].abs()

        # Compute the value of the Manual Portfolio
        manual_strategy = marketsimcode.compute_portvals(orders_df, start_val = sv, commission=0.00, impact=0.00)
        
        # Normalize
        manual_prices = manual_strategy['Value']
        manual_prices = manual_prices/manual_prices[0]
        manual_strategy['Value'] = manual_prices
        manual_strategy.rename(columns={'Value':'Manual Strategy Portfolio'}, inplace=True)



        # Compute the value of the Benchmark Portfolio
        benchmark_orders = pd.DataFrame({'Symbol': symbol, 'Order':'BUY', 'Shares':1000}, index = [prices_all.index[0], prices_all.index[-1]])
        benchmark = marketsimcode.compute_portvals(benchmark_orders, start_val = sv, commission=0.00, impact=0.00)
      
        # Normalize
        bench_prices = benchmark['Value']
        bench_prices = bench_prices/bench_prices[0]
        benchmark['Value'] = bench_prices
        benchmark.rename(columns={'Value':'Benchmark Portfolio'}, inplace=True)


        # Get long and short dates
        buy_dates = orders_df.index[orders_df['Order'] == 'BUY']
        sell_dates = orders_df.index[orders_df['Order'] == 'SELL']

        """
        # Plot
        if not OOS:
            ax = manual_strategy.plot(title='Figure 5: Benchmark and Manual Strategy Portfolio', color = 'r')
        else:
            ax = manual_strategy.plot(title='Figure 6: Out of Sample Benchmark and Manual Strategy Portfolio', color = 'r')
        benchmark.plot(ax=ax, color = 'g')
        ax.set_xlabel('Date', fontsize = 12)
        ax.set_ylabel('Normalized Price', fontsize = 12)
        if not OOS:
            for bd in buy_dates:
                ax.axvline(bd, color='b')
            for sd in sell_dates:
                ax.axvline(sd, color='k')
        if not OOS:
            plt.savefig('Figure_5')
        else:
            plt.savefig('Figure_6')

        """
        
        return df_trades


def author():  		   	  			  	 		  		  		    	 		 		   		 		  
    return 'atorreshotrum3'


if __name__ == "__main__":                                                                                        
    ms = ManualStrategy()

    # In sample
    df_trades = ms.testPolicy(symbol = "JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv = 100000)
    
    # Out of sample
    df_trades = ms.testPolicy(symbol = "JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011,12,31), sv = 100000)
    