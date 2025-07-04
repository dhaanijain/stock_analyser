import yfinance as yf
import pandas as pd
import re
import datetime
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for script environments
import matplotlib.pyplot as plt
from gnews import GNews
from datetime import timedelta, date
from krazy import postgres_utilities as pu
from general import connect_to_db
from sqlalchemy.sql import text
from krazy.utility_functions import LoggerConfigurator
from krazy import postgres_utilities as pu
import numpy as np
from general import logger, upsert_postgres_from_dataframe


engine = connect_to_db()
cur = engine.connect()


# Parameters
TICKER = "AAPL"
START_DATE = "2025-04-01"  
END_DATE = "2025-06-12"     # Keep end date as before
QUERY = "Apple"
GNEWS_API_KEY = "abe68bc5c29ee58b31c7ebbcb07b1290"  # Updated API key


# 1. Fetch stock data
def stock_price_fetch_data(ticker, start_date, end_date):
    """
    Fetch historical stock price data for a given ticker and date range using yfinance.
    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
    Returns:
        pd.DataFrame: DataFrame with stock price data and additional columns.
    """
    df_price = yf.download(ticker, start=start_date, end=end_date)
    if df_price is None or not isinstance(df_price, pd.DataFrame) or df_price.empty:
        logger.error(f"No data returned for ticker {ticker} from {start_date} to {end_date}.")
        return pd.DataFrame()
    # Flatten MultiIndex columns if present, keep only the first value
    if isinstance(df_price.columns, pd.MultiIndex):
        df_price.columns = [col[0] if isinstance(col, tuple) else col for col in df_price.columns.values]
    df_price.reset_index(inplace=True)
    if 'Date' in df_price.columns:
        df_price['date'] = pd.to_datetime(df_price['Date'])
    else:
        df_price['date'] = df_price.index.date
    df_price['time_stamp'] = df_price['date']  # new column for time_stamp
    df_price['stock_code'] = ticker
    df_price['unique_key'] = df_price['stock_code'] + ':' + df_price['date'].astype(str)
    logger.debug('Stock prices fetched successfully')
    return df_price

# 2.push stock data to database
def stock_price_push_data(df_price):
    """
    Push stock price DataFrame to the database using upsert.
    Args:
        df_price (pd.DataFrame): DataFrame with stock price data.
    Returns:
        None
    """
    if df_price is None or df_price.empty:
        logger.warning('No stock price data to push to database.')
        return

    upsert_postgres_from_dataframe(
        engine,
        df_price,
        'stock_analyzer',
        'stock_prices',
        'unique_key',
        mode="upsert",      # or "insert"/"update" as needed
        add_cols=True,       # set as needed
        alter_cols=True      # set as needed
    )
    logger.debug('Stock prices pushed to database successfully')
    
    
# 4. Fetch all articles using GNews    
def all_articles_fetch_data(query, start_dt, end_dt):
    """
    Fetch all news articles for a given query and date range using GNews.
    Args:
        query (str): Search query (usually ticker or company name).
        start_dt (str or date): Start date.
        end_dt (str or date): End date.
    Returns:
        pd.DataFrame or None: DataFrame with articles, or None if no articles found.
    """
    
    all_articles = []

    def daterange(start_dt, end_dt, step_days=30):
        for n in range(0, (end_dt - start_dt).days, step_days):
            yield start_dt + timedelta(n), min(start_dt + timedelta(n + step_days), end_dt)

    start_dt = pd.to_datetime(start_dt).date()
    end_dt = pd.to_datetime(end_dt).date()
    logger.debug(f'Fetching articles for query: {query} from {start_dt} to {end_dt}')
    for start, end in daterange(start_dt, end_dt, step_days=30):
        gnews = GNews(language='en', max_results=100, start_date=start, end_date=end)
        articles = gnews.get_news(query)
        all_articles.extend(articles)
    all_articles = pd.DataFrame(all_articles)
    if all_articles.empty==False:
        all_articles.rename(columns={'title':'headline'}, inplace=True)
        if 'publisher' in all_articles.columns:
            all_articles['link'] = all_articles['publisher'].apply(lambda x: dict(x).get('href'))
            all_articles['title'] = all_articles['publisher'].apply(lambda x: dict(x).get('title'))

        
        try:
            all_articles['published date'] = pd.to_datetime(all_articles['published date'])
        except Exception as e:
            all_articles['published date'] = pd.to_datetime(all_articles['published date'],format='%a, %d %b %Y %H:%M:%S GMT')    
        
        
        all_articles['date'] = all_articles['published date'].dt.date
        all_articles['stock_code'] = query
        all_articles.drop(columns=['publisher'], inplace=True)
        all_articles.rename(columns={'published date':'published_date'}, inplace=True)
            
        logger.debug('All articles fetched successfully')
        
        return all_articles
    else:
        return None

# 5. Push all articles to database
def all_articles_push_data(all_articles):
    """
    Push all articles DataFrame to the database using upsert.
    Args:
        all_articles (pd.DataFrame): DataFrame with news articles.
    Returns:
        None
    """
    all_articles['unique_key'] = all_articles['headline'].astype(str) + '_' + \
                                all_articles['date'].astype(str) + '_' + \
                                all_articles['stock_code'].astype(str)
    all_articles['sentiment_score'] = np.nan
    upsert_postgres_from_dataframe(
        engine,
        all_articles,
        'stock_analyzer',
        'stock_articles',
        'unique_key',
        mode="upsert",      # or "insert"/"update" as needed
        add_cols=True,       # set as needed
        alter_cols=True      # set as needed
    )
    logger.debug('All articles pushed to database successfully')

# 6. pull stock prices from database    
def fetch_prices(ticker:str, start_date:str, end_date:str)->pd.DataFrame:
    """
    Pull stock prices from the database for a given ticker and date range.
    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
    Returns:
        pd.DataFrame: DataFrame with stock price data.
    """
    df = pd.read_sql_query(f'''select * from stock_analyzer.stock_prices sp where sp.stock_code ~* 
                           '{ticker}' and sp."time_stamp" between '{start_date}' and '{end_date}';''', engine)    
    return df

# 7. pull articles from database
def articles_fetch_data(ticker:str, start_date:str, end_date:str)->pd.DataFrame:
    """
    Pull news articles from the database for a given ticker and date range.
    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
    Returns:
        pd.DataFrame: DataFrame with news articles.
    """
    df_articles = pd.read_sql_query(f'''select * from stock_analyzer.stock_articles sa 
                                    where sa.stock_code ~* '{ticker}' 
                                    and sa."published_date" between '{start_date}' and '{end_date}';''', engine)
    return df_articles


