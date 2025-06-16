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
    print(f"Number of articles fetched for sentiment analysis: {len(df_articles)}")
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
        print(f"Error in pushing sentiment score: {e}")
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
    # df_merged.to_sql(
    #     'temp_table_stock_prices',
    #     engine,
    #     if_exists='replace',
    #     index=False, schema='stock_analyzer'
    # )
    logger.debug('Updating sentiment score in stock prices table')
    pu.dbase_writer_dup_handled(
        engine,
        df_merged,
        'stock_analyzer',
        'stock_prices',
        'row_id',
        files_processed=None,
        update_dup=True
    )
    logger.debug('Stock prices updated with article count and sentiment score successfully')
    