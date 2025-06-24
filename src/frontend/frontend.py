"""
frontend.py - STOXiE.ai Streamlit Frontend

This file implements the Streamlit-based frontend for the STOXiE.ai project. It provides a user-friendly interface for:
- Inputting stock ticker, date range, and analysis options
- Triggering the backend data pipeline
- Fetching processed data from the database
- Displaying AI recommendations, summary metrics, news articles, and interactive visualizations

Key Features:
- Sidebar for user input
- Main area for results, charts, and recommendations
- Robust error handling and user feedback
- Modular integration with backend and recommendation engine

Typical usage: Run this file with Streamlit to launch the web app.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import subprocess
import sqlite3



# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from data_handeling import (
    stock_price_fetch_data, 
    stock_price_push_data,
    all_articles_fetch_data,
    all_articles_push_data,
    fetch_prices
)

from recommendation import recommendation_generator

from general import connect_to_db
from data_processing import sentiment_analysis, merge_tables
from data_analysis import plot_area_chart, plot_sentiment_heatmap_plotly, plot_gradient_sentiment_overlay, plot_interactive, plot_sentiment_spikes, predict_stock_price_for_ticker_date_df, compare_models

# Page config
st.set_page_config(
    page_title="STOXiE.ai",
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
    st.title("STOXiE.ai")
    
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
        with st.spinner("Running backend data pipeline..."):
            # Run main.py to fetch/process data and update DB
            result = subprocess.run([
                sys.executable, os.path.join(os.path.dirname(__file__), '..', 'backend', 'main.py')
            ], capture_output=True, text=True)
            if result.returncode != 0:
                st.error(f"Backend error: {result.stderr}")
                st.stop()
            else:
                st.success("Data pipeline completed. Fetching results...")
        
        with st.spinner("Fetching processed data from database..."):
            # Connect to SQLite database (adjust path if needed)
            db_path = os.path.join(os.path.dirname(__file__), '..', 'backend', 'stock_data.db')
            # conn = sqlite3.connect(db_path)
            conn = connect_to_db()
            # Fetch price data
            query_price = f"""
                SELECT * FROM "stock_analyzer".stock_prices WHERE stock_code = '{ticker}' AND "date" BETWEEN '{start_date}' AND '{end_date}' ORDER BY date;
            """
            df_price = pd.read_sql_query(query_price, conn)
            
            try:
                compare_models(df_price)
            except Exception as e:
                st.warning(f"Model training failed: {e}")
            future_date = (pd.to_datetime(end_date) + pd.Timedelta(days=15)).date()
            predicted_price = predict_stock_price_for_ticker_date_df(df_price, ticker, str(future_date))
            # Ensure predicted_price is a native Python float for all downstream usage
            if hasattr(predicted_price, 'item'):
                predicted_price = predicted_price.item()
            else:
                predicted_price = float(predicted_price)
            
            if not df_price.empty:
                st.success("Stock price data loaded from database!")
                
            else:
                st.error("No stock price data available for the selected period.")
                st.stop()
            # Fetch articles if needed
            if include_articles:
                query_articles = f"""
                    SELECT * FROM stock_analyzer.stock_articles WHERE stock_code = '{ticker}' AND "published_date" BETWEEN '{start_date}' AND '{end_date}' ORDER BY published_date DESC
                """
                df_articles = pd.read_sql_query(query_articles, conn)
                if not df_articles.empty:
                    st.success("News articles loaded from database!")
                else:
                    st.warning("No news articles found for the selected period.")
                result = recommendation_generator(df_price, df_articles, predicted_price)
            else:
                result = recommendation_generator(df_price, pd.DataFrame(), predicted_price)
                    
            
            # conn.close()
            
            # if include_articles:
            #     with st.spinner("Fetching and processing news articles..."):
            #         # df_articles = all_articles_fetch_data(ticker, start_date, end_date)
            #         if not df_articles.empty:
            #             all_articles_push_data(df_articles)
            #             st.success("News articles processed successfully!")
            #         else:
            #             st.warning("No news articles found for the selected period.")
                
            #     with st.spinner("Analyzing sentiment..."):
            #         sentiment_analysis(ticker, start_date, end_date)
            #         merge_tables(ticker, start_date, end_date)
            #         st.success("Sentiment analysis completed!")
            
            # Display results
            # Display AI Recommendation at the top
            st.markdown("## AI Recommendation")
            st.info(result)

            st.title(f"{ticker} Stock Analysis")

            # Ensure models are trained before prediction
            

            # Predict and display the predicted price for 15 days after the selected end date
            try:
                # Ensure predicted_price is a native Python float
                predicted_price = float(predicted_price)
                st.metric(f"Predicted Price ({future_date})", f"${predicted_price:.2f}")
            except Exception as e:
                st.warning(f"Prediction unavailable: {e}")

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
            
            # # Price Chart
            # st.subheader("Stock Price")
            # st.line_chart(df_price.set_index('date')['Close'])
            
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
            # if include_sentiment:
            #     st.subheader("Sentiment Analysis")
            #     try:
            #         # Get the latest data from database
            #         df_price = fetch_prices(ticker, start_date, end_date)
            #         if not df_price.empty and 'sentiment_score' in df_price.columns:
            #             fig = plot_area_chart(df_price)
            #             st.plotly_chart(fig, use_container_width=True)
            #         else:
            #             st.warning("No sentiment analysis data available for visualization.")
            #     except Exception as e:
            #         st.warning(f"Could not generate sentiment analysis visualization: {str(e)}")
            
            
            # Price Chart
            st.subheader("Stock Price vs Sentiment Area Chart")
            fig_area = plot_area_chart(df_price)
            st.plotly_chart(fig_area, use_container_width=True, key="area_chart")

            # Sentiment Heatmap
            st.subheader("Sentiment Score Heatmap by Day")
            fig_heatmap = plot_sentiment_heatmap_plotly(df_price)
            st.plotly_chart(fig_heatmap, use_container_width=True, key="heatmap")
            
            df_price.sort_values(by='date', inplace=True, ascending=True)
            # df_price = df_price.loc[df_price['sentiment_score'].notna()]
            
            # Gradient Sentiment Overlay
            st.subheader("Stock Price with Sentiment Intensity Overlay")
            fig_gradient = plot_gradient_sentiment_overlay(df_price)
            st.plotly_chart(fig_gradient, use_container_width=True, key="gradient_overlay")

            # Interactive Stock vs Sentiment Trend
            st.subheader("Interactive Stock vs Sentiment Trend")
            fig_interactive = plot_interactive(df_price)
            st.plotly_chart(fig_interactive, use_container_width=True, key="interactive")

            # Sentiment Spikes
            st.subheader("Sentiment Spikes Over Time")
            fig_spikes = plot_sentiment_spikes(df_price)
            st.plotly_chart(fig_spikes, use_container_width=True, key="spikes")

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
