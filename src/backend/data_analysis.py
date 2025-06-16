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




# def visualize_stock_prices(df_price):
#     import matplotlib.pyplot as plt

#     df_price['date'] = pd.to_datetime(df_price['date'])
#     df_price.set_index('date', inplace=True)

#     # Smooth the sentiment score for trend clarity
#     df_price['sentiment_score_smooth'] = df_price['sentiment_score'].rolling(window=3).mean()

#     fig, ax1 = plt.subplots(figsize=(14, 7))

#     # Plot stock price
#     ax1.plot(df_price.index, df_price['Close'], label='Stock Price', color='blue')
#     ax1.set_ylabel('Stock Price', color='blue')
#     ax1.tick_params(axis='y', labelcolor='blue')

#     # Plot article count if exists
#     if 'article_count' in df_price.columns:
#         df_price['article_count'].plot(kind='bar', ax=ax1, alpha=0.2, width=0.5, label='Article Count', color='gray')

#     # Plot sentiment on second Y-axis
#     ax2 = ax1.twinx()
#     if 'sentiment_score_smooth' in df_price.columns:
#         ax2.plot(df_price.index, df_price['sentiment_score_smooth'], label='Sentiment Score (Smoothed)', color='orange')
#     else:
#         print("sentiment_score column not found.")

#     ax2.set_ylabel('Sentiment Score', color='orange')
#     ax2.tick_params(axis='y', labelcolor='orange')

#     # Labels, legend, grid
#     plt.title('Historical Stock Price vs Sentiment Trend')
#     fig.tight_layout()
#     fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

#     return fig

    
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
