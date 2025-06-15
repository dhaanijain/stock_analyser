# All code from the notebook, adapted for script use
import yfinance as yf
import pandas as pd
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
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

# Parameters
NEWS_API_KEY = "abe68bc5c29ee58b31c7ebbcb07b1290"
TICKER = "AAPL"
START_DATE = "2023-06-12"   # 2 years before today
END_DATE = "2025-06-11"     # Keep end date as before
QUERY = "Apple"

# 1. Fetch stock data
df_price = yf.download(TICKER, start=START_DATE, end=END_DATE)
if isinstance(df_price.columns, pd.MultiIndex):
    df_price.columns = df_price.columns.get_level_values(0)
df_price['pct_change'] = df_price['Close'].pct_change() * 100
df_price.reset_index(inplace=True)
df_price['date'] = pd.to_datetime(df_price['Date']).dt.date

# 2. Fetch news headlines using GNews (instead of NewsAPI)
gnews = GNews(language='en', country='US', max_results=100)
all_articles = []

def daterange(start_date, end_date, step_days=30):
    for n in range(0, (end_date - start_date).days, step_days):
        yield start_date + timedelta(n), min(start_date + timedelta(n + step_days), end_date)

start_dt = pd.to_datetime(START_DATE).date()
end_dt = pd.to_datetime(END_DATE).date()

for start, end in daterange(start_dt, end_dt, step_days=30):
    articles = gnews.get_news(QUERY)
    for article in articles:
        published = article.get('published date')
        if hasattr(published, 'date'):
            art_date = published.date()
        elif isinstance(published, str):
            try:
                art_date = pd.to_datetime(published).date()
            except Exception:
                art_date = None
        else:
            art_date = None
        if art_date and (start <= art_date <= end):
            all_articles.append(article)

data = []
for article in all_articles:
    title = article.get('title', '')
    published = article.get('published date')
    if hasattr(published, 'date'):
        date_val = published.date()
    elif isinstance(published, str):
        try:
            date_val = pd.to_datetime(published).date()
        except Exception:
            date_val = None
    else:
        date_val = None
    data.append({'date': date_val, 'headline': title})
df_sentiment = pd.DataFrame(data)
df_sentiment = df_sentiment.dropna(subset=['date'])
df_sentiment['date'] = pd.to_datetime(df_sentiment['date']).dt.date

# 3. Sentiment analysis on headlines
analyzer = SentimentIntensityAnalyzer()
df_sentiment['sentiment_score'] = df_sentiment['headline'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
def label_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'
df_sentiment['sentiment'] = df_sentiment['sentiment_score'].apply(label_sentiment)

# 4. Merge sentiment with stock % change on date
df_merged = pd.merge(df_sentiment, df_price[['date', 'pct_change']], on='date', how='inner')

# 5. Correlation
corr = df_merged['sentiment_score'].corr(df_merged['pct_change'])
print(f"Correlation between sentiment score and price % change: {corr:.3f}")

# 6. Create binary label for pct_change: 1 if pct_change > 0, else 0
df_merged['pct_change_label'] = (df_merged['pct_change'] > 0).astype(int)

# 7. One-hot encode sentiment
encoder = OneHotEncoder(sparse_output=False, dtype=int)
sentiment_arr = encoder.fit_transform(df_merged[['sentiment']])
sentiment_df = pd.DataFrame(sentiment_arr, columns=encoder.get_feature_names_out(['sentiment']), index=df_merged.index)
df_merged = pd.concat([df_merged, sentiment_df], axis=1)

# 8. Visualization: Daily Avg Sentiment Score vs. Stock Price % Change
daily = df_merged.groupby('date').agg({'sentiment_score': 'mean', 'pct_change': 'mean'}).reset_index()
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.bar(daily['date'], daily['sentiment_score'], color='royalblue', label='Avg Sentiment Score', alpha=0.7)
ax1.set_ylabel('Avg Sentiment Score', color='royalblue')
ax1.set_xlabel('Date')
ax1.tick_params(axis='y', labelcolor='royalblue')
ax2 = ax1.twinx()
ax2.plot(daily['date'], daily['pct_change'], color='firebrick', marker='o', label='Stock % Change')
ax2.set_ylabel('Stock % Change', color='firebrick')
ax2.tick_params(axis='y', labelcolor='firebrick')
plt.title('Daily Avg Sentiment Score (bar) vs. Stock Price % Change (line)')
fig.tight_layout()
fig.autofmt_xdate()
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
plt.savefig('sentiment_vs_stock.png')
plt.close()
print('Saved plot: sentiment_vs_stock.png')

# 9. Train and Evaluate RandomForest & Logistic Regression Models
sentiment_cols = [col for col in df_merged.columns if col.startswith('sentiment_')]
X = df_merged[sentiment_cols]
y = df_merged['pct_change_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
lr = LogisticRegression(max_iter=200, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print('RandomForest Classification Report:')
print(classification_report(y_test, y_pred_rf))
print('Logistic Regression Classification Report:')
print(classification_report(y_test, y_pred_lr))
print(f'RandomForest Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}')
print(f'Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr):.3f}')
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test, ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title('RandomForest Confusion Matrix')
ConfusionMatrixDisplay.from_estimator(lr, X_test, y_test, ax=axes[1], cmap='Greens', colorbar=False)
axes[1].set_title('Logistic Regression Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.close()
print('Saved plot: confusion_matrices.png')

# 10. Visualize Correct vs Wrong Predictions
results = X_test.copy()
results['True Label'] = y_test.values
results['RF Prediction'] = y_pred_rf
results['LR Prediction'] = y_pred_lr
results['RF Correct'] = results['True Label'] == results['RF Prediction']
results['LR Correct'] = results['True Label'] == results['LR Prediction']

plt.figure(figsize=(8, 4))
sns.countplot(x='RF Correct', data=results, palette=['salmon', 'mediumseagreen'])
plt.title('RandomForest: Correct vs Wrong Predictions')
plt.xlabel('Prediction Correct?')
plt.ylabel('Count')
plt.xticks([0, 1], ['Wrong', 'Correct'])
plt.savefig('rf_correct_wrong.png')
plt.close()
print('Saved plot: rf_correct_wrong.png')

plt.figure(figsize=(8, 4))
sns.countplot(x='LR Correct', data=results, palette=['salmon', 'mediumseagreen'])
plt.title('Logistic Regression: Correct vs Wrong Predictions')
plt.xlabel('Prediction Correct?')
plt.ylabel('Count')
plt.xticks([0, 1], ['Wrong', 'Correct'])
plt.savefig('lr_correct_wrong.png')
plt.close()
print('Saved plot: lr_correct_wrong.png')




