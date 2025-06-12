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
import matplotlib.pyplot as plt

# Parameters
NEWS_API_KEY = "a7f3930774df422f89548bc580f2abea"
TICKER = "AAPL"
START_DATE = "2025-05-11"
END_DATE = "2025-06-10"
QUERY = "Apple"

# 1. Fetch stock data
df_price = yf.download(TICKER, start=START_DATE, end=END_DATE)
if isinstance(df_price.columns, pd.MultiIndex):
    df_price.columns = df_price.columns.get_level_values(0)
df_price['pct_change'] = df_price['Close'].pct_change() * 100
df_price.reset_index(inplace=True)
df_price['date'] = pd.to_datetime(df_price['Date']).dt.date

# 2. Fetch news headlines from NewsAPI
newsapi = NewsApiClient(api_key=NEWS_API_KEY)
all_articles = newsapi.get_everything(q=QUERY, from_param=START_DATE, to=END_DATE, language='en', sort_by='relevancy', page_size=100)
articles = all_articles['articles']
data = []
for article in articles:
    title = article['title']
    date = article['publishedAt'][:10]
    data.append({'date': date, 'headline': title})
df_sentiment = pd.DataFrame(data)
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
plt.show()

# 9. Train and Evaluate RandomForest & Logistic Regression Models
sentiment_cols = [col for col in df_merged.columns if col.startswith('sentiment_')]
X = df_merged[sentiment_cols]
y = df_merged['pct_change_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.85, random_state=42)
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
plt.show()

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
plt.show()
plt.figure(figsize=(8, 4))
sns.countplot(x='LR Correct', data=results, palette=['salmon', 'mediumseagreen'])
plt.title('Logistic Regression: Correct vs Wrong Predictions')
plt.xlabel('Prediction Correct?')
plt.ylabel('Count')
plt.xticks([0, 1], ['Wrong', 'Correct'])
plt.show()




