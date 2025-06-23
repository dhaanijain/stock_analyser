from crewai import Agent, Task, Crew, LLM, Process
from dotenv import load_dotenv
import os
import pandas as pd
load_dotenv()

def recommendation_generator(df_price:pd.DataFrame, df_articles:pd.DataFrame):
    
    llm_4o_mini = LLM(
        model = 'openai/gpt-4o-mini',
        api_key=os.getenv('API_KEY') # enter API key here
    )

    agent = Agent(
        goal='Generating stock recommendation',
        role='Generating stock recommendation for user on the basis of the stock anaysis.',
        backstory='You are a very stock analyst who is very experienced in this stream and you help me generate a recommendation for the user on the basis of the stock anaysis. For example:- If the user should hold, buy or sell the stock according to the condition of the stock at the moment.',
        llm=llm_4o_mini
    )

    task = Task(
        description='Generate a stock recommendation based on the data provided. \n Price Data: {price_data} \n Top 10 articles: {articles_data}',
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
        'articles_data': df_articles.head(10).to_string(index=False)
    })

    return response