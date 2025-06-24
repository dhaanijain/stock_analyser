from data_handeling import articles_fetch_data, fetch_prices
from general import connect_to_db
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from general import logger
import pandas as pd
from krazy import postgres_utilities as pu
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from general import upsert_postgres_from_dataframe

engine = connect_to_db()
cur = engine.connect()

#1. sentiment analysis on stock articles
def sentiment_analysis(ticker:str, start_date:str, end_date:str)->bool:
    """
    Perform sentiment analysis on news articles for a given ticker and date range.
    Updates the sentiment_score in the database for each article.
    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
    Returns:
        bool: True if successful, False otherwise.
    """
    df_articles = pd.read_sql_query(f'''select * 
                    from 
                        stock_analyzer.stock_articles sa 
                    where 
                        sa.stock_code ~* '{ticker}' 
                        and sa."published_date" between '{start_date}' 
                        and '{end_date}' 
                        and sa.sentiment_score is null or sa.sentiment_score = 'NaN';
                    ''', engine)
    print(f"Number of articles fetched for sentiment analysis: {len(df_articles)}")
    # do sentimant analysis on df_articles and save the score in column 'sentiment_score'
    analyzer = SentimentIntensityAnalyzer()
    df_articles['sentiment_score'] = df_articles['headline'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    try:
        upsert_postgres_from_dataframe(
            engine,
            df_articles[['row_id', 'sentiment_score']],
            'stock_analyzer',
            'stock_articles',
            'row_id',
            mode="upsert",      # or "insert"/"update" as needed
            add_cols=True,       # set as needed
            alter_cols=True      # set as needed
        )
        # pu.dbase_writer_dup_handled(
        #     engine,
        #     df_articles[['sentiment_score', 'row_id']],
        #     'stock_analyzer',
        #     'stock_articles',
        #     'row_id',
        #     files_processed=None,
        #     update_dup=True
        # )
        return True
    except Exception as e:
        print(f"Error in pushing sentiment score: {e}")
        logger.error(f"Error in pushing sentiment score: {e}")
        return False

#2. group by all articles
def groupby_all_articles(ticker:str, start_date:str, end_date:str)->pd.DataFrame:
    """
    Group articles by stock_code and date, aggregating count and mean sentiment.
    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
    Returns:
        pd.DataFrame: DataFrame with article_count and mean sentiment per day.
    """
    df_articles = articles_fetch_data(ticker, start_date, end_date)
    df_articles = df_articles.groupby(['stock_code','date']).agg({'headline': 'count', 'sentiment_score': 'mean'}).reset_index()
    df_articles.rename(columns={'headline': 'article_count'}, inplace=True)
    return df_articles

#3. merge stock prices and articles
def merge_tables(ticker:str, start_date:str, end_date:str)->pd.DataFrame:
    """
    Merge stock prices and grouped articles, calculate pct_change, and upsert to DB.
    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
    Returns:
        pd.DataFrame: Merged DataFrame with sentiment, article_count, and pct_change.
    """
    df_articles = groupby_all_articles(ticker, start_date, end_date).reset_index()
    df_prices = fetch_prices(ticker, start_date, end_date)
    df_prices.drop(columns=[
        'sentiment_score', 'article_count', 'pct_change'  # Drop pct_change if present
    ], inplace=True, errors='ignore')
    # Calculate pct_change
    df_prices = df_prices.sort_values(by='date')
    df_prices['pct_change'] = df_prices['Close'].pct_change()
    df_merged = pd.merge(df_prices, df_articles, on=['stock_code','date'], how='left').reset_index(drop=True)
    df_merged['sentiment_score'] = df_merged['sentiment_score'].astype(float)
    df_merged['article_count'] = df_merged['article_count'].astype(float)
    logger.debug('Updating sentiment score and pct_change in stock prices table')
    upsert_postgres_from_dataframe(
        engine,
        df_merged,
        'stock_analyzer',
        'stock_prices',
        'row_id',
        mode="upsert",      # or "insert"/"update" as needed
        add_cols=True,       # set as needed
        alter_cols=True      # set as needed
    )
    logger.debug('Stock prices updated with article count, sentiment score, and pct_change successfully')
    
# do one hot encoding on df_prices on sentiment column

def one_hot_encode_sentiment(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Perform one-hot encoding on the sentiment_score column of a DataFrame.
    Args:
        df_prices (pd.DataFrame): DataFrame with a 'sentiment_score' column.
    Returns:
        pd.DataFrame: DataFrame with one-hot encoded sentiment columns added.
    """
    if 'sentiment' not in df_prices.columns:
        print("Sentiment column not found in DataFrame.")
        return df_prices
    
    encoder = OneHotEncoder(sparse_output=False, dtype=int)
    sentiment_arr = encoder.fit_transform(df_prices[['sentiment_score']])
    sentiment_df = pd.DataFrame(sentiment_arr, columns=encoder.get_feature_names_out(['sentiment_score']), index=df_prices.index)
    
    df_prices = pd.concat([df_prices, sentiment_df], axis=1)

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
