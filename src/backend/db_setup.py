"""
db_setup.py

This script sets up the PostgreSQL database schema and tables required for the STOXiE.ai stock analyzer application using SQLAlchemy ORM.

Features:
- Defines ORM models for stock_articles and stock_prices tables under the stock_analyzer schema
- Supports creation of the schema if it does not exist (with --create-schema flag)
- Can be run as a script to initialize the database for the app

Usage:
    python db_setup.py --create-schema  # Creates schema and tables
    python db_setup.py                  # Creates tables only (if schema exists)

Environment variables required (in .env):
    DB_USERNAME, DB_PASSWORD, DB_HOST, DB_PORT, DB_DATABASE
"""

from sqlalchemy import create_engine, Column, Integer, BigInteger, String, Date, Float, Text, TIMESTAMP, text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
import argparse

# Load environment variables from .env file
load_dotenv()

DB_USER = os.getenv('DB_USERNAME')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_DATABASE')

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

Base = declarative_base()

class StockArticles(Base):
    __tablename__ = 'stock_articles'
    __table_args__ = {'schema': 'stock_analyzer'}
    row_id = Column(Integer, primary_key=True, nullable=False)
    headline = Column(Text)
    description = Column(Text)
    published_date = Column(TIMESTAMP)
    url = Column(Text)
    link = Column(Text)
    title = Column(String(50))
    date = Column(Date)
    stock_code = Column(String(10))
    unique_key = Column(Text)
    sentiment_score = Column(Float)

class StockPrices(Base):
    __tablename__ = 'stock_prices'
    __table_args__ = {'schema': 'stock_analyzer'}
    row_id = Column(Integer, primary_key=True, nullable=False)
    date = Column(Date)  # was Date
    Close = Column(Float)
    High = Column(Float)
    Low = Column(Float)
    Open = Column(Float)
    Volume = Column(BigInteger)
    runtime = Column(TIMESTAMP)
    pct_change = Column(Float)
    time_stamp = Column(Date)  # was date
    stock_code = Column(String(10))
    unique_key = Column(Text)
    sentiment_score = Column(Float)  # new column
    article_count = Column(Float)    # new column

def create_schema_and_tables(create_schema=False):
    engine = create_engine(DATABASE_URL)
    if create_schema:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE SCHEMA IF NOT EXISTS stock_analyzer;
            """))
    Base.metadata.create_all(engine)
    print('Database tables created successfully.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set up the stock_analyzer database tables.')
    parser.add_argument('--create-schema', action='store_true', help='Create the stock_analyzer schema if it does not exist.')
    args = parser.parse_args()
    create_schema_and_tables(create_schema=args.create_schema)
