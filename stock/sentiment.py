# %%
import yfinance as yf
import pandas as pd
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# === Parameters ===
NEWS_API_KEY = "a7f3930774df422f89548bc580f2abea"
TICKER = "AAPL"
START_DATE = "2025-05-10"
END_DATE = "2025-06-09"
QUERY = "Apple"

# --- Utility Functions ---
def clean_text(text):
    text = re.sub(r"http\\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9 ]+", "", text)
    return text.lower()

def label_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# --- 1. Fetch stock data ---
def fetch_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df['pct_change'] = df['Close'].pct_change() * 100
    df.reset_index(inplace=True)
    df['date'] = pd.to_datetime(df['Date']).dt.date
    return df[['date', 'pct_change']]

# --- 2. Fetch news headlines ---
def fetch_news_headlines(api_key, query, start, end):
    newsapi = NewsApiClient(api_key=api_key)
    all_articles = newsapi.get_everything(q=query,
                                          from_param=start,
                                          to=end,
                                          language='en',
                                          sort_by='relevancy',
                                          page_size=100)
    articles = all_articles.get('articles', [])
    data = []
    for article in articles:
        title = article.get('title', '')
        date = article.get('publishedAt', '')[:10]
        if title and date:
            data.append({'date': date, 'headline': title})
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df

# --- 3. Sentiment analysis ---
def analyze_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()
    df['clean_headline'] = df['headline'].apply(clean_text)
    df['sentiment_score'] = df['clean_headline'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    df['sentiment'] = df['sentiment_score'].apply(label_sentiment)
    return df

# --- 4. Merge and correlate ---
def merge_and_correlate(df_sentiment, df_price):
    df_merged = pd.merge(df_sentiment, df_price, on='date', how='inner')
    corr = df_merged['sentiment_score'].corr(df_merged['pct_change'])
    print(f"\nCorrelation between sentiment score and price % change: {corr:.3f}")
    return df_merged[['date', 'headline', 'sentiment', 'pct_change', 'sentiment_score']]

# --- Main Execution ---
def main():
    df_price = fetch_stock_data(TICKER, START_DATE, END_DATE)
    df_news = fetch_news_headlines(NEWS_API_KEY, QUERY, START_DATE, END_DATE)
    if df_news.empty:
        print("No news articles found for the given period.")
        return
    df_sentiment = analyze_sentiment(df_news)
    df_merged = merge_and_correlate(df_sentiment, df_price)
    df_merged.to_csv('cleaned_headlines_sentiment.csv', index=False)
    print(df_merged.head())

    # --- Visualization ---
    # Aggregate by date
    daily = df_merged.groupby('date').agg({
        'sentiment_score': 'mean',
        'pct_change': 'mean'
    }).reset_index()
    daily['date'] = pd.to_datetime(daily['date'])

    # Set up the matplotlib figure
    fig, ax1 = plt.subplots(figsize=(12, 6))
    # Bar for average sentiment score per day
    ax1.bar(daily['date'], daily['sentiment_score'], color='royalblue', label='Avg Sentiment Score', alpha=0.7)
    ax1.set_ylabel('Avg Sentiment Score', color='royalblue')
    ax1.set_xlabel('Date')
    ax1.tick_params(axis='y', labelcolor='royalblue')

    # Create a second y-axis for stock % change
    ax2 = ax1.twinx()
    ax2.plot(daily['date'], daily['pct_change'], color='firebrick', marker='o', label='Stock % Change')
    ax2.set_ylabel('Stock % Change', color='firebrick')
    ax2.tick_params(axis='y', labelcolor='firebrick')

    # Add title and legends
    plt.title('Daily Avg Sentiment Score (bar) vs. Stock Price % Change (line)')
    fig.tight_layout()
    fig.autofmt_xdate()
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    plt.show()




if __name__ == "__main__":
    main()




