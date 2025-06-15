from sqlalchemy import create_engine
from sqlalchemy.engine import URL, Engine
from sqlalchemy.sql import text
from dotenv import load_dotenv
from krazy.utility_functions import LoggerConfigurator
import os

# Load environment variables from .env file
load_dotenv()

def connect_to_db():
    # Get credentials from environment variables
    username = os.getenv("DB_USERNAME")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", 5432)
    database = os.getenv("DB_DATABASE")

    # Create the connection string
    url = URL.create(
        drivername="postgresql",
        username=username,
        password=password,
        host=host,
        port=port,
        database=database
    )
    engine = create_engine(url)
    return engine

# initialize logging
logger = LoggerConfigurator('stock_analyzer',
                            r'C:\Users\dhaan\OneDrive\Documents\Python\AI Project\logs',
                            'DEBUG').configure()