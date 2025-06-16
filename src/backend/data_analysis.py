#write a function to visualize the historical data of the table stock_prices with columns date semntiment_score and article_count

from datetime import timedelta
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sqlalchemy import create_engine, text
from krazy import postgres_utilities as pu
from general import connect_to_db
from general import logger
import matplotlib.pyplot as plt
from data_handeling import fetch_prices
from data_processing import one_hot_encode_sentiment
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px

    
def data_analysis():
    ticker = "AAPL"
    start_date = "2025-05-12"
    end_date = "2025-06-11"
    df_price = fetch_prices(ticker, start_date, end_date)
    
    # df_price = one_hot_encode_sentiment(df_price)
    fig = plot_area_chart(df_price)
    # fig2 = price_analysis(df_price)
    # if fig2 is not None:
    #     fig2.savefig(f"data_analysis_{ticker}_{start_date}_{end_date}.png")
    logger.info("Data analysis completed successfully.")
    return fig
    # return fig2


# ticker = "AAPL"
# start_date = "2025-05-12"
# end_date = "2025-06-11"
# fig = data_analysis()
# fig.savefig(f"data_analysis_{ticker}_{start_date}_{end_date}.png")

# 1. Heatmap Calendar of Sentiment Scores
def plot_sentiment_heatmap_plotly(df_price):

    df = df_price.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month_name()
    # Only use months present in the data
    months_present = df['month'].unique().tolist()
    # Create the pivot table
    pivot = df.pivot_table(index='month', columns='day', values='sentiment_score', aggfunc='mean')
    # Reindex to only present months, sorted by calendar order
    months_order = pd.date_range('2023-01-01', periods=12, freq='MS').strftime('%B').tolist()
    months_sorted = [m for m in months_order if m in months_present]
    pivot = pivot.reindex(months_sorted)
    fig = px.imshow(
        pivot,
        color_continuous_scale='RdYlGn',
        aspect='auto',
        labels=dict(color='Sentiment Score'),
        title='Sentiment Score Heatmap by Day'
    )
    fig.update_xaxes(title='Day')
    fig.update_yaxes(title='Month')
    fig.show()
    return fig

# Usage:
# fig = plot_sentiment_heatmap_plotly(df_price)




# 2. Area Chart of Close Price and Sentiment
import plotly.graph_objs as go

#TODO: fix y axis left and right scale to exclude Nan values and set min value to 10% below the minimum value of the series
def plot_area_chart(df_price):
    import pandas as pd
    df = df_price.copy()
    df['date'] = pd.to_datetime(df['date'])
    full_range = pd.date_range(df['date'].min(), df['date'].max())
    df = df.set_index('date').reindex(full_range).reset_index().rename(columns={'index': 'date'})
    df.set_index('date', inplace=True)
    print("running plot_area_chart (plotly version)")

    # Exclude NaN values for min calculation
    min_close = df['Close'].dropna().min()
    yaxis_min = min_close - 0.1 * abs(min_close) if pd.notnull(min_close) else 0

    min_sentiment = df['sentiment_score'].dropna().min()
    yaxis2_min = min_sentiment - 0.1 * abs(min_sentiment) if pd.notnull(min_sentiment) else 0

    fig = go.Figure()

    # Stock Close Price (Primary Y-axis)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'],
        mode='lines+markers',
        name='Close Price',
        line=dict(color='royalblue', width=3, shape='spline'),
        marker=dict(size=6, color='royalblue', symbol='circle'),
        fill='tozeroy',
        fillcolor='rgba(65, 105, 225, 0.15)',
        yaxis='y1',
        hovertemplate='Date: %{x}<br>Close: %{y:.2f}<extra></extra>',
        connectgaps=False
    ))

    # Sentiment Score (Secondary Y-axis)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['sentiment_score'],
        mode='lines+markers',
        name='Sentiment Score',
        line=dict(color='orange', width=3, shape='spline', dash='dot'),
        marker=dict(size=6, color='orange', symbol='diamond'),
        fill='tozeroy',
        fillcolor='rgba(255, 165, 0, 0.15)',
        yaxis='y2',
        hovertemplate='Date: %{x}<br>Sentiment: %{y:.2f}<extra></extra>',
        connectgaps=False
    ))

    fig.update_layout(
        title=dict(
            text="Stock Price vs Sentiment Area Chart",
            font=dict(size=22, color='black'),
            x=0.5
        ),
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor='rgba(200,200,200,0.2)',
            tickangle=45
        ),
        yaxis=dict(
            title="Close Price",
            showgrid=True,
            gridcolor='rgba(200,200,255,0.2)',
            side='left',
            range=[yaxis_min, None]
        ),
        yaxis2=dict(
            title="Sentiment Score",
            overlaying='y',
            side='right',
            showgrid=False,
            range=[yaxis2_min, None]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
        autosize=False,
        width=950,
        height=520,
        margin=dict(l=60, r=60, t=80, b=80)
    )
    fig.show()
    return fig

# 3. Dual Line with Sentiment Gradient Background
import plotly.graph_objs as go
import numpy as np

import plotly.graph_objs as go

def plot_gradient_sentiment_overlay(df_price):
    df = df_price.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    sentiment = df['sentiment_score'].rolling(3, min_periods=1).mean()

    # Normalize sentiment for color mapping
    norm_sentiment = (sentiment - sentiment.min()) / (sentiment.max() - sentiment.min())
    colors = [f'rgba({int(255*(1-s))},{int(255*s)},0,0.15)' for s in norm_sentiment]

    fig = go.Figure()

    # Add Close Price (primary y-axis)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'],
        mode='lines+markers',
        name='Close Price',
        line=dict(color='blue', width=2),
        yaxis='y1'
    ))

    # Add Smoothed Sentiment (secondary y-axis) as area fill
    fig.add_trace(go.Scatter(
        x=df.index, y=sentiment,
        mode='lines',
        name='Smoothed Sentiment',
        line=dict(color='orange', width=2, dash='dot'),
        fill='tozeroy',
        fillcolor='rgba(255,165,0,0.15)',
        yaxis='y2'
    ))

    # Add colored background rectangles for sentiment
    for i in range(len(df)-1):
        fig.add_vrect(
            x0=df.index[i], x1=df.index[i+1],
            fillcolor=colors[i], opacity=0.5, line_width=0
        )

    fig.update_layout(
        title="Stock Price with Sentiment Intensity Overlay",
        xaxis_title="Date",
        yaxis=dict(
            title="Close Price",
            showgrid=True,
            side='left'
        ),
        yaxis2=dict(
            title="Smoothed Sentiment",
            overlaying='y',
            side='right',
            showgrid=False
        ),
        template="plotly_white",
        width=950,
        height=520
    )
    fig.show()
    return fig

# 4. Interactive Plot with Plotly
import plotly.graph_objs as go

def plot_interactive(df_price):
    import pandas as pd
    fig = go.Figure()

    # Ensure date is datetime and fill missing dates
    df = df_price.copy()
    df['date'] = pd.to_datetime(df['date'])
    full_range = pd.date_range(df['date'].min(), df['date'].max())
    df = df.set_index('date').reindex(full_range).reset_index().rename(columns={'index': 'date'})

    # Add Close Price line (primary y-axis)
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['Close'],
        mode='lines+markers',
        name='Close Price',
        line=dict(color='royalblue', width=2),
        yaxis='y1',
        connectgaps=False
    ))

    # Add Sentiment Score line (secondary y-axis)
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['sentiment_score'],
        mode='lines+markers',
        name='Sentiment Score',
        line=dict(color='orange', width=2, dash='dot'),
        yaxis='y2',
        connectgaps=False
    ))

    fig.update_layout(
        title='Interactive Stock vs Sentiment Trend',
        xaxis_title='Date',
        yaxis=dict(
            title='Close Price',
            showgrid=True,
            side='left'
        ),
        yaxis2=dict(
            title='Sentiment Score',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        legend_title='Legend',
        template='plotly_white',
        width=900,
        height=500
    )
    fig.show()
    return fig

# 5. Sentiment Spike Detector
import plotly.graph_objs as go

def plot_sentiment_spikes(df_price):
    df = df_price.copy()
    df['sentiment_diff'] = df['sentiment_score'].diff()
    spikes = df[abs(df['sentiment_diff']) > 0.5]

    fig = go.Figure()

    # Sentiment Score line
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['sentiment_score'],
        mode='lines+markers',
        name='Sentiment Score',
        line=dict(color='blue', width=2)
    ))

    # Spikes as red markers
    fig.add_trace(go.Scatter(
        x=spikes['date'],
        y=spikes['sentiment_score'],
        mode='markers',
        name='Spikes',
        marker=dict(color='red', size=10, symbol='diamond')
    ))

    fig.update_layout(
        title="Sentiment Spikes Over Time",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        legend_title="Legend",
        template="plotly_white",
        width=900,
        height=450
    )
    fig.show()
    return fig

# write a function to train random forest model on the stock_prices table with columns date, sentiment_score, article_count and Close price
from sklearn.ensemble import RandomForestRegressor
def train_random_forest_model(df_price):
    df = df_price.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Prepare features and target
    X = df[['sentiment_score', 'article_count']]
    y = df['Close']

    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Feature importance
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # add accuracy score
    accuracy = model.score(X, y)
    print(f"Model Accuracy: {accuracy:.2f}")
    
    #add f1 score
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y, model.predict(X))
    print(f"Mean Squared Error: {mse:.2f}")

    print("Model trained successfully.")
    print("Feature Importances:")
    print(importance_df)

    return model, importance_df, mse, accuracy

def train_logistic_regression_model(df_price):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    df = df_price.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Prepare features and target
    x = df[['sentiment_score', 'article_count']]
    y = (df['Close'] > df['Close'].shift(1)).astype(int)  # Binary target: 1 if price increased, else 0

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train Logistic Regression model
    model = LogisticRegression(random_state=42)
    X_train = X_train.dropna()
    y_train = y_train.loc[X_train.index]  # keep y_train aligned
    model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)

    return model, accuracy, report


def train_xgboost_model(df_price):
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import accuracy_score

    df = df_price.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Prepare features and target
    X = df[['sentiment_score', 'article_count']]
    y = df['Close']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost model
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    # accuracy = model.score(X_test, y_test)
    # accuracy = accuracy_score(y_test, y_pred)
    
    print(f"XGBoost Model Mean Squared Error: {mse:.2f}")

    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred)
    print(f"XGBoost Model R^2 Score: {r2:.2f}")

    return model, mse, r2

    
#write the code to compare models and select the best one based on accuracy and mean squared error by calling the functions above
def compare_models(df_price):
    print("Training Logistic Regression Model...")
    lr_model, lr_accuracy, lr_report = train_logistic_regression_model(df_price)
    
    print("\nTraining Random Forest Model...")
    rf_model, rf_importance, rf_mse, rf_accuracy = train_random_forest_model(df_price)
    
    print("\nTraining XGBoost Model...")
    xgb_model, xgb_mse, xgb_r2 = train_xgboost_model(df_price)

    # Compare models based on accuracy and MSE
    print("\nModel Comparison:")
    print(f"Logistic Regression - Accuracy: {lr_accuracy:.2f}")
    print(f"Random Forest - Accuracy: {rf_accuracy:.2f}, MSE: {rf_mse:.2f}")
    print(f"XGBoost - R^2 Score: {xgb_r2:.2f}, MSE: {xgb_mse:.2f}")

    # Select the best model based on criteria
    best_model = None
    if rf_accuracy > lr_accuracy and rf_mse < xgb_mse:
        best_model = "Random Forest"
    elif xgb_r2 > rf_accuracy and xgb_mse < rf_mse:
        best_model = "XGBoost"
    else:
        best_model = "Logistic Regression"

    print(f"\nBest Model Selected: {best_model}")

    # Save all trained models for later use
    import joblib
    import os
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(lr_model, os.path.join(models_dir, 'logistic_regression_model.pkl'))
    joblib.dump(rf_model, os.path.join(models_dir, 'random_forest_model.pkl'))
    joblib.dump(xgb_model, os.path.join(models_dir, 'xgboost_model.pkl'))

def predict_stock_price_for_ticker_date_df(df_price, ticker, date):
    """
    Predict the stock price for a given ticker and date using the best available model.
    If the date is not present in df_price, use the most recent available data for prediction.
    """
    import joblib
    import os
    import pandas as pd

    df = df_price.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['ticker'] == ticker] if 'ticker' in df.columns else df
    df.set_index('date', inplace=True)

    # If the date is not in the DataFrame, use the most recent available row
    if pd.to_datetime(date) in df.index:
        X = df.loc[[pd.to_datetime(date)], ['sentiment_score', 'article_count']]
    else:
        # Use the latest available data for the ticker
        latest_row = df.sort_index().iloc[[-1]][['sentiment_score', 'article_count']]
        X = latest_row
        print(f"Date {date} not found for ticker {ticker}. Using latest available data from {latest_row.index[0].strftime('%Y-%m-%d')} for prediction.")

    # Determine the best model (same logic as compare_models)
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    model_paths = [
        ('xgboost_model.pkl', 'XGBoost'),
        ('random_forest_model.pkl', 'Random Forest'),
        ('logistic_regression_model.pkl', 'Logistic Regression')
    ]
    model = None
    model_name = None
    for fname, name in model_paths:
        path = os.path.join(models_dir, fname)
        if os.path.exists(path):
            model = joblib.load(path)
            model_name = name
            break
    if model is None:
        raise FileNotFoundError("No trained model found. Please run compare_models first.")

    # Predict using the selected model
    prediction = model.predict(X)
    print(f"Predicted by {model_name} model for ticker {ticker}.")
    return prediction[0]  # Return the predicted price


