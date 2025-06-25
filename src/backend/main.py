"""
main.py - STOXiE.ai Backend Pipeline Entrypoint

This script orchestrates the backend data pipeline for the STOXiE.ai project. It performs the following steps:

1. Fetches stock price data for a given ticker and date range
2. Pushes the price data to the database
3. Fetches news articles for the ticker and date range
4. Pushes the articles to the database (if any are found)
5. Runs sentiment analysis on the news articles
6. Merges sentiment and article data with price data, calculates pct_change, and updates the database

Typical usage: Run this script as the backend data pipeline, either standalone or triggered by the frontend.
"""

import pandas as pd
from data_handeling import all_articles_fetch_data, stock_price_fetch_data
from data_handeling import all_articles_push_data, stock_price_push_data
from data_processing import sentiment_analysis
from data_processing import merge_tables
from data_analysis import data_analysis
from general import logger
import time

ticker = "AAPL"
start_date = "2025-05-12"
end_date = "2025-06-11"

def main():

    logger.debug('Fetching prices')
    df_price = stock_price_fetch_data(ticker, start_date, end_date)
    stock_price_push_data(df_price)
    # stock_price_fetch_data_if_not_exists(ticker, start_date, end_date)

    logger.debug('Getting articles')
    df_articles = all_articles_fetch_data(ticker, start_date, end_date)
    if df_articles is not None:
        all_articles_push_data(df_articles)
    else:
        print("No articles found for the given date range.")
        
    logger.debug('Running sentiment analysis')
    sentiment_analysis(ticker, start_date, end_date)
    
    logger.debug('Combining stock prices and articles')
    merge_tables(ticker, start_date, end_date) #includes call to groupby_all_articles
    
    # print(df_merged)
    print("run successfully")
    # data_analysis()
    
    
    
    
if __name__ == '__main__':
    
    # sentiment_analysis("AAPL", "2025-05-12", "2025-06-11")
    main()