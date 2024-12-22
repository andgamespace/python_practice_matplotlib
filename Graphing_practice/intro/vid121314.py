import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import mplfinance as mpf
import numpy as np
import pandas as pd
import yfinance as yf
from mplfinance.original_flavor import candlestick_ohlc



def graph_data(stock):
    fig = plt.figure()
    fig, ax1 = plt.subplots(figsize=(10, 6))

    stock_data = yf.Ticker(stock)
    df = stock_data.history(period='10y')
    dates = df.index  
    if df.empty or 'Close' not in df.columns or 'High' not in df.columns or 'Low' not in df.columns:
        print(f"No valid data available for {stock}")
    print(df)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

    
    
    ax1.plot(df.index, df['High'], label='High Prices', color='blue', alpha=0.7)
    ax1.plot(df.index, df['Low'], label='Low Prices', color='red', alpha=0.7)
    plt.xticks(rotation=45)
    '''for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(45)
        This is an alternative to the above line which does it 
        in one line'''
    #The above line is how to rotate the x-labels by 45 degrees, being more space-efficient, so when you make the 
    #window smaller, it still is readable.
    #ax1.fill_between(df.index, df['High'], df['Low'], color='gray', alpha=0.3, label='Price Range')
    ax1.grid(True, color='g', linestyle='-')
    #linewidth is another option for the grid
    
    ax1.xaxis.label.set_color('c')
    ax1.yaxis.label.set_color('r')
    '''ax1.set_yticks([0, 25, 50, 75])
    the above line(now commented out) is how you change the 
    yticks to be what you set them as'''
    

    plt.fill_between(df.index, df['High'], df['Low'], color='gray', alpha=0.3, label='Price Range')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('stock')
    plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)
    plt.legend()
    plt.show()
    mpf.plot(stock_data, type='candle', style='style')

graph_data('MSFT')
