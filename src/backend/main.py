import pandas as pd
from data_handeling import all_articles_fetch_data, stock_price_fetch_data
from data_handeling import all_articles_push_data, stock_price_push_data
from data_processing import sentiment_analysis
from data_processing import merge_tables
from data_handeling import stock_price_fetch_data_if_not_exists
from data_processing import one_hot_encode_sentiment
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
    all_articles_push_data(df_articles)
    
    logger.debug('Running sentiment analysis')
    sentiment_analysis(ticker, start_date, end_date)
    
    logger.debug('Combining stock prices and articles')
    merge_tables(ticker, start_date, end_date) #includes call to groupby_all_articles
    #TODO: Fix merge not working when previous functions are uncommented
    
    # print(df_merged)
    print("run successfully")
    # data_analysis()
    
    
    
    
if __name__ == '__main__':
    
    # sentiment_analysis("AAPL", "2025-05-12", "2025-06-11")
    main()