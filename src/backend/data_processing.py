from data_handeling import articles_fetch_data, fetch_prices
from general import connect_to_db
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from general import logger
import pandas as pd
from krazy import postgres_utilities as pu

engine = connect_to_db()
cur = engine.connect()

#1. sentiment analysis on stock articles
def sentiment_analysis(ticker:str, start_date:str, end_date:str)->bool:
    df_articles = pd.read_sql_query(f'''select * 
                    from 
                        stock_analyzer.stock_articles sa 
                    where 
                        sa.stock_code ~* '{ticker}' 
                        and sa."date" between '{start_date}' 
                        and '{end_date}' 
                        and sa.sentiment_score is null;
                    ''', engine)
    # do sentimant analysis on df_articles and save the score in column 'sentiment_score'
    analyzer = SentimentIntensityAnalyzer()
    df_articles['sentiment_score'] = df_articles['headline'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    try:
        pu.dbase_writer_dup_handled(
            engine,
            df_articles[['sentiment_score', 'row_id']],
            'stock_analyzer',
            'stock_articles',
            'row_id',
            files_processed=None,
            update_dup=True
        )
        return True
    except Exception as e:
        logger.error(f"Error in pushing sentiment score: {e}")
        return False

#2. group by all articles
def groupby_all_articles(ticker:str, start_date:str, end_date:str)->pd.DataFrame:
    df_articles = articles_fetch_data(ticker, start_date, end_date)
    df_articles = df_articles.groupby(['stock_code','date']).agg({'headline': 'count', 'sentiment_score': 'mean'}).reset_index()
    df_articles.rename(columns={'headline': 'article_count'}, inplace=True)
    return df_articles

#3. merge stock prices and articles
def merge_tables(ticker:str, start_date:str, end_date:str)->pd.DataFrame:
    df_articles = groupby_all_articles(ticker, start_date, end_date)
    df_prices = fetch_prices(ticker, start_date, end_date)
    df_merged = pd.merge(df_prices, df_articles, on=['stock_code','date'], how='left')
    return df_merged