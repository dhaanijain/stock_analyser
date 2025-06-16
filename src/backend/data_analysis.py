#write a function to visualize the historical data of stock prices
from datetime import timedelta
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sqlalchemy import create_engine, text
from krazy import postgres_utilities as pu
from general import connect_to_db
from general import logger

def visualize_stock_prices(df_price):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(14, 7))
    plt.plot(df_price['date'], df_price['Close'], label='Close Price', color='blue')
    plt.title('Stock Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()