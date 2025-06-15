import yfinance as yf
import pandas as pd
import re
import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
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
from general import logger

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
    df_price = yf.download(ticker, start=start_date, end=end_date)
    df_price['runtime'] = datetime.datetime.now()
    if isinstance(df_price.columns, pd.MultiIndex):
        df_price.columns = df_price.columns.get_level_values(0)
    df_price['pct_change'] = df_price['Close'].pct_change() * 100
    df_price.reset_index(inplace=True)
    df_price['date']  = pd.to_datetime(df_price['Date']).dt.date
    # df_price['stock_name']=stock_name
    df_price['stock_code']=ticker
    logger.debug('Stock prices fectched successfully')
    return df_price

#2.push stock data to database
def stock_price_push_data(df_price):
    
    df_price['unique_key'] = df_price['Date'].astype(str)
    pu.dbase_writer_dup_handled(
        engine,
        df_price,
        'stock_analyzer',
        'stock_prices',
        'unique_key',
        files_processed=None,
        update_dup=True
    )
    # df_price.to_sql('stock_prices', con=engine, if_exists='append', index=False)
    logger.debug('Stock prices pushed to database successfully')
    
    
# 3. Fetch all articles using GNews    
def all_articles_fetch_data(query, start_dt, end_dt):
    
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
    all_articles.rename(columns={'title':'headline'}, inplace=True)
    all_articles['link'] = all_articles['publisher'].apply(lambda x: dict(x).get('href'))
    all_articles['title'] = all_articles['publisher'].apply(lambda x: dict(x).get('title'))
    
    try:
        all_articles['published date'] = pd.to_datetime(all_articles['published date'])
    except Exception as e:
        all_articles['published date'] = pd.to_datetime(all_articles['published date'],format='%a, %d %b %Y %H:%M:%S GMT')    
    
    
    all_articles['date'] = all_articles['published date'].dt.date
    all_articles['stock_code'] = query
    all_articles.drop(columns=['publisher'], inplace=True)
        
    logger.debug('All articles fetched successfully')
    return all_articles

# 4. Push all articles to database
def all_articles_push_data(all_articles):
    
    all_articles['unique_key'] = all_articles['date'].astype(str) 
    all_articles['sentiment_score'] = np.nan
    pu.dbase_col_checker_adder("stock_analyzer", 'stock_articles', all_articles, engine, speak=True)
    pu.dbase_writer_dup_handled(
        engine,
        all_articles,
        'stock_analyzer',
        'stock_articles',
        'unique_key',
        files_processed=None,
        update_dup=True
    )
    logger.debug('All articles pushed to database successfully')

#5. pull stock prices from database    
def fetch_prices(ticker:str, start_date:str, end_date:str)->pd.DataFrame:
    df = pd.read_sql_query(f'''select * from stock_analyzer.stock_prices sp where sp.stock_code ~* 
                           '{ticker}' and sp."date" between '{start_date}' and '{end_date}';''', engine)    
    return df

#6. pull articles from database
def articles_fetch_data(ticker:str, start_date:str, end_date:str)->pd.DataFrame:
    df_articles = pd.read_sql_query(f'''select * from stock_analyzer.stock_articles sa 
                                    where sa.stock_code ~* '{ticker}' 
                                    and sa."date" between '{start_date}' and '{end_date}';''', engine)
    return df_articles