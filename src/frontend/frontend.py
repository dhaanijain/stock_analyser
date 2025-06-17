import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from data_handeling import (
    stock_price_fetch_data, 
    stock_price_push_data,
    all_articles_fetch_data,
    all_articles_push_data,
    fetch_prices
)
from data_processing import sentiment_analysis, merge_tables
from data_analysis import plot_area_chart

# Page config
st.set_page_config(
    page_title="Stock Analyzer",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #3F51B5;
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar inputs
with st.sidebar:
    st.title("Stock Analyzer")
    
    # Stock Input
    ticker = st.text_input("Stock Ticker", "AAPL").upper()
    
    # Date Range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            datetime.now() - timedelta(days=30)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            datetime.now()
        )
    
    # Analysis Options
    st.subheader("Analysis Options")
    include_sentiment = st.checkbox("Include Sentiment Analysis", value=True)
    include_articles = st.checkbox("Include News Articles", value=True)
    
    # Analyze Button
    analyze_button = st.button("Analyze Stock", type="primary")

# Main content area
if analyze_button:
    try:
        with st.spinner("Fetching and processing stock data..."):
            # Fetch and push stock data
            df_price = stock_price_fetch_data(ticker, start_date, end_date)
            if not df_price.empty:
                stock_price_push_data(df_price)
                st.success("Stock price data processed successfully!")
            else:
                st.error("No stock price data available for the selected period.")
                st.stop()
            
            if include_articles:
                with st.spinner("Fetching and processing news articles..."):
                    df_articles = all_articles_fetch_data(ticker, start_date, end_date)
                    if not df_articles.empty:
                        all_articles_push_data(df_articles)
                        st.success("News articles processed successfully!")
                    else:
                        st.warning("No news articles found for the selected period.")
                
                with st.spinner("Analyzing sentiment..."):
                    sentiment_analysis(ticker, start_date, end_date)
                    merge_tables(ticker, start_date, end_date)
                    st.success("Sentiment analysis completed!")
            
            # Display results
            st.title(f"{ticker} Stock Analysis")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"${df_price['Close'].iloc[-1]:.2f}")
            with col2:
                price_change = df_price['Close'].iloc[-1] - df_price['Close'].iloc[0]
                st.metric("Price Change", f"${price_change:.2f}")
            with col3:
                if include_articles and 'df_articles' in locals():
                    st.metric("Articles Analyzed", len(df_articles))
            with col4:
                if include_sentiment and 'sentiment_score' in df_price.columns:
                    avg_sentiment = df_price['sentiment_score'].mean()
                    st.metric("Avg Sentiment", f"{avg_sentiment:.2f}")
                else:
                    st.metric("Avg Sentiment", "N/A")
            
            # Price Chart
            st.subheader("Stock Price")
            st.line_chart(df_price.set_index('date')['Close'])
            
            # Articles Table
            if include_articles and 'df_articles' in locals():
                st.subheader("Recent News Articles")
                if 'sentiment_score' in df_articles.columns:
                    display_columns = ['headline', 'published_date', 'sentiment_score']
                else:
                    display_columns = ['headline', 'published_date']
                st.dataframe(
                    df_articles[display_columns].head(10),
                    hide_index=True
                )
            
            # Sentiment Analysis
            if include_sentiment:
                st.subheader("Sentiment Analysis")
                try:
                    # Get the latest data from database
                    df_price = fetch_prices(ticker, start_date, end_date)
                    if not df_price.empty and 'sentiment_score' in df_price.columns:
                        fig = plot_area_chart(df_price)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No sentiment analysis data available for visualization.")
                except Exception as e:
                    st.warning(f"Could not generate sentiment analysis visualization: {str(e)}")
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    # Welcome message
    st.title("Welcome to Stock Analyzer")
    st.write("""
    Please use the sidebar to:
    1. Enter a stock ticker
    2. Select date range
    3. Choose analysis options
    4. Click 'Analyze Stock' to begin
    """)
