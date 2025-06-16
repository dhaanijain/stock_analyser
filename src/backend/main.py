import pandas as pd
from data_handeling import all_articles_fetch_data, stock_price_fetch_data
from data_handeling import all_articles_push_data, stock_price_push_data
from data_processing import sentiment_analysis
from data_processing import merge_tables
import time


def main():
    ticker = "AAPL"
    start_date = "2025-05-12"
    end_date = "2025-06-11"
    
    # df_price = stock_price_fetch_data(ticker, start_date, end_date)
    # stock_price_push_data(df_price)

    
    # df_articles = all_articles_fetch_data(ticker, start_date, end_date)
    # all_articles_push_data(df_articles)
    
    # time.sleep(5)  # Ensure the data is pushed before sentiment analysis
    
    # sentiment_analysis(ticker, start_date, end_date)
    
    df_merged = merge_tables(ticker, start_date, end_date) #includes call to groupby_all_articles
    print(df_merged)
    print("run successfully")
    
    
    
if __name__ == '__main__':
    
    # sentiment_analysis("AAPL", "2025-05-12", "2025-06-11")
    main()