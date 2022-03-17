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
import matplotlib.ticker as ticker



def author():  		   	  			  	 		  		  		    	 		 		   		 		  
    return 'atorreshotrum3'   		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  
def b_band(symbol, dates):
    prices_all = get_data([symbol], dates)  # automatically adds SPY  
    del prices_all['SPY']

    SMA = prices_all[symbol].rolling(window=20).mean()

    # Calculate STDev
    rstd = prices_all[symbol].rolling(window=20).std()	

    # Calculate upper and lower bands
    upper = SMA + 2 * rstd
    lower = SMA - 2 * rstd

    # Calculate %B
    percent_B = ((prices_all[symbol] - lower) / (upper - lower)) * 100
    
    """
    # BB plot
    prices_all['Upper Band'] = upper
    prices_all['Lower Band'] = lower
    prices_all['SMA'] = SMA

    ax = plt.subplot(211)
    ax.plot(prices_all['Upper Band'], label= 'Upper Band')
    ax.plot(prices_all['Lower Band'], label= 'Lower Band')
    ax.plot(prices_all['SMA'], label= 'SMA')
    ax.legend(loc="lower right")
    plt.title('Figure 1: JPM Bollinger Bands and BBP')
    ax.fill_between(prices_all.index, lower, upper, color='lightskyblue', alpha='0.3')
    ax.set_ylabel('Price', fontsize = 12)
    #ax.set_xlim([prices_all.index[0], prices_all.index[-1]])
    plt.setp(ax.get_xticklabels(), visible=False)

    ax2 = plt.subplot(212, sharex=ax)
    ax2.set_xlabel('Date', fontsize = 12)
    ax2.set_ylabel('Percentage', fontsize = 12)
    ax2.fill_between(prices_all.index, 0, 100, color='lightcoral', alpha='0.3')
    plt.xticks(rotation=20)

    plt.plot(percent_B, label='BBP')
    ax2.legend(loc="upper left")
    plt.savefig('Figure_1')

    plt.clf()
    plt.cla()
    plt.close()
    """

    return percent_B



def RSI(symbol, dates):
    prices_all = get_data([symbol], dates)  # automatically adds SPY  
    del prices_all['SPY']

    # Set standard period
    period=14

    changes = prices_all[symbol].diff()

    gains = pd.DataFrame(changes.copy())
    gains[gains < 0] = 0
    gains['rolling'] = gains[symbol].rolling(period).mean()

    losses = pd.DataFrame(changes.copy())
    losses[losses > 0] = 0
    losses['rolling'] = losses[symbol].rolling(period).mean().abs()

    rsi = 100 - 100 / (1 + gains['rolling'] / losses['rolling'])  

    prices_all['RSI'] = rsi

    """
    ax3 = plt.subplot(211)
    plt.plot(prices_all[symbol], label=symbol)
    ax3.legend(loc="lower left")
    plt.title('Figure 2: JPM Price and RSI')
    #ax.set_xlabel('Date', fontsize = 12)
    ax3.set_ylabel('Price', fontsize = 12)
    ax3.set_xlim([prices_all.index[0], prices_all.index[-1]])
    plt.setp(ax3.get_xticklabels(), visible=False)

    ax4 = plt.subplot(212, sharex=ax3)
    ax4.set_xlabel('Date', fontsize = 12)
    ax4.set_ylabel('RSI', fontsize = 12)
    plt.axhline(y=70, color ='b')
    plt.axhline(y=30, color ='r')
    plt.xticks(rotation=40)
    plt.plot(prices_all['RSI'],  color='g', label="RSI")
    ax4.legend(loc="upper left")
    plt.savefig('Figure_2')
    plt.clf()
    plt.cla()
    plt.close()
    """

    rsi_df = prices_all['RSI']

    return rsi_df


def MACD(symbol, dates):
    prices_all = get_data([symbol], dates)  # automatically adds SPY  
    del prices_all['SPY']

    first_exp = prices_all[symbol].ewm(span=12, adjust=False).mean()
    second_exp = prices_all[symbol].ewm(span=26, adjust=False).mean()
    prices_all['MACD'] = first_exp-second_exp
    prices_all['Signal'] = prices_all['MACD'].ewm(span=9, adjust=False).mean()
    prices_all['diff'] = prices_all['MACD'] - prices_all['Signal']


    """
    ax5 = plt.subplot(211)
    plt.plot(prices_all[symbol], label=symbol)
    ax5.legend(loc="upper left")
    plt.title('Figure 3: JPM Price and MACD')
    ax5.set_ylabel('Price', fontsize = 12)
    ax5.set_xlim([prices_all.index[0], prices_all.index[-1]])
    plt.setp(ax5.get_xticklabels(), visible=False)

    ax6 = plt.subplot(212, sharex=ax5)
    ax6.set_xlabel('Date', fontsize = 12)
    ax6.set_ylabel('MACD Value', fontsize = 12)
    plt.plot(prices_all['MACD'], label='MACD', color = 'y')
    plt.plot(prices_all['Signal'], label='Signal Line', color='c')
    plt.plot(prices_all['diff'], label='Difference', color='r')
    plt.axhline(y=0, color ='b', linestyle = '--')
    plt.legend(loc='upper left')
    plt.xticks(rotation=40)
    ax6.xaxis.set_major_locator(ticker.MultipleLocator(7))
    plt.savefig('Figure_3')

    plt.clf()
    plt.cla()
    plt.close()
    """

    diff_df = prices_all['diff']

    return diff_df




if __name__ == "__main__":

    # Set date range
    in_sample_dates = pd.date_range(start='1/1/2008', end='12/31/2009')	  
    out_of_sample_dates = pd.date_range(start='1/1/2010', end='12/31/2011')	
    symbol = 'JPM'  

    bbp_df = b_band(symbol, in_sample_dates)	 

    rsi_df = RSI(symbol, in_sample_dates)		  		  		
    
    #Solid in sample dates for MACD
    in_sample_dates = pd.date_range(start='1/1/2008', end='3/1/2008')	      	 		 		   		 		  
    diff_df = MACD(symbol, in_sample_dates)
    
    