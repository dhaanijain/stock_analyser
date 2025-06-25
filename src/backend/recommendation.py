"""
recommendation.py

This module provides the AI-powered stock recommendation engine for the STOXiE.ai project.
It uses CrewAI and an LLM (OpenAI GPT-4o-mini) to generate actionable stock recommendations
(buy, hold, sell) based on price and news article data.

Key Features:
- Loads API key from environment variables
- Formats price and article data for the LLM
- Defines an agent and task for CrewAI
- Returns a natural language recommendation for the user

Typical usage: Call recommendation_generator(df_price, df_articles) from the frontend or backend.
"""

from crewai import Agent, Task, Crew, LLM, Process
from dotenv import load_dotenv
import os
import pandas as pd
load_dotenv()

def recommendation_generator(df_price:pd.DataFrame, df_articles:pd.DataFrame, predicted_price:float) -> str:
    
    llm_4o_mini = LLM(
        model = 'openai/gpt-4o-mini',
        api_key=os.getenv('API_KEY') # enter API key here
    )

    agent = Agent(
        goal='Generating stock recommendation',
        role='Generating stock recommendation for user on the basis of the stock anaysis.',
        backstory='You are a very stock analyst who is very experienced in this stream and you help me generate a recommendation for the user on the basis of the stock anaysis and predicted price for the date 15 days after the end date. For example:- If the user should hold, buy or sell the stock according to the condition of the stock at the moment.',
        llm=llm_4o_mini
    )

    task = Task(
        description='Generate a stock recommendation based on the data provided. \n Price Data: {price_data} \n Top 10 articles: {articles_data} \n {predicted_price}',
        expected_output='stock recommendation for the user',
        agent=agent
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True,
        process=Process.sequential
    )

    response = crew.kickoff(inputs={
        'price_data': df_price[['date', 'Close', 'High', 'Low', 'Open' , 'Volume', 'pct_change', 'stock_code', 'sentiment_score']].to_string(index=False),
        'articles_data': df_articles.head(10).to_string(index=False),
        'predicted_price': predicted_price
        
    })

    return response