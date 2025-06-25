from sqlalchemy import create_engine
from sqlalchemy.engine import URL, Engine
from sqlalchemy.sql import text
from dotenv import load_dotenv
from krazy.utility_functions import LoggerConfigurator
from sqlalchemy import Table, MetaData, update, String, Float, Integer, Date, TIMESTAMP, text
from sqlalchemy.dialects.postgresql import insert
import os
import numpy as np
from sqlalchemy import inspect
import pandas as pd
from typing import Optional

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
        port=port, # type: ignore
        database=database
    )
    engine = create_engine(url)
    return engine

def insert_dataframe_to_postgres(engine, df, table_name, schema, columns):
    """
    Insert a DataFrame into a PostgreSQL table using SQLAlchemy.
    - engine: SQLAlchemy engine
    - df: pandas DataFrame
    - table_name: str, name of the table
    - schema: str, schema name
    - columns: list of column names to insert
    """
    metadata = MetaData(schema=schema)
    table = Table(table_name, metadata, autoload_with=engine)
    data = df[columns].to_dict(orient='records')
    stmt = insert(table).values(data)
    with engine.begin() as conn:
        conn.execute(stmt)

def update_postgres_from_dataframe(engine, df, table_name, schema, key_column):
    """
    Update rows in a PostgreSQL table using a DataFrame and a key column.
    - engine: SQLAlchemy engine
    - df: pandas DataFrame
    - table_name: str, name of the table
    - schema: str, schema name
    - key_column: str, column name to match for updates
    """
    metadata = MetaData(schema=schema)
    table = Table(table_name, metadata, autoload_with=engine)
    with engine.begin() as conn:
        for _, row in df.iterrows():
            update_dict = row.drop(key_column).to_dict()
            stmt = (
                update(table)
                .where(getattr(table.c, key_column) == row[key_column])
                .values(**update_dict)
            )
            conn.execute(stmt)

def upsert_postgres_from_dataframe(
    engine: Engine,
    df: pd.DataFrame,
    schema: str,
    table_name: str,
    key_column: Optional[str] = None,
    mode: str = "upsert",
    add_cols: bool = False,
    alter_cols: bool = False
) -> bool:
    """
    Flexible DataFrame-to-Postgres sync function.
    Returns True if operation is successful, else returns False and prints/logs the error.
    Parameters:
        engine: SQLAlchemy engine
        df: pandas DataFrame
        schema: str, schema name
        table_name: str, table name
        key_column: str or None, column to match for upsert/insert/update. If None, only insert is allowed.
        mode: 'upsert' (default), 'insert', or 'update'
        add_cols: bool, if True add missing columns to DB, else skip them (default: False)
        alter_cols: bool, if True alter column types/lengths to fit DataFrame, else skip (default: False)
    Behavior:
        - If key_column is None, only insert is allowed (mode is forced to 'insert').
        - 'upsert': Upsert if key_column is unique/PK, else insert only new rows. Respects add_cols/alter_cols.
        - 'insert': Insert only new rows (ignore existing). Respects add_cols/alter_cols.
        - 'update': Update only existing rows (ignore new). Respects add_cols/alter_cols.
    """
    try:
        print(f"[INFO] Starting sync to {schema}.{table_name} (mode={mode}, add_cols={add_cols}, alter_cols={alter_cols}, key_column={key_column})")
        # Drop index column if present
        if df.index.name is not None or 'index' in df.columns or 'level_0' in df.columns:
            df = df.reset_index(drop=True)
            for idx_col in ['index', 'level_0']:
                if idx_col in df.columns:
                    df = df.drop(columns=[idx_col])
        metadata = MetaData(schema=schema)
        table = Table(table_name, metadata, autoload_with=engine)
        db_columns = {col.name: col for col in table.columns}
        df_columns = set(df.columns)
        missing_columns = df_columns - set(db_columns.keys())
        dtype_map = {
            'object': String,
            'float64': Float,
            'int64': Integer,
            'datetime64[ns]': Date,
            'bool': Integer,
            'float32': Float,
            'int32': Integer,
            'datetime64[ns, UTC]': TIMESTAMP
        }
        # Add missing columns if requested
        if add_cols:
            print(f"[INFO] Adding missing columns: {missing_columns}")
            for col in missing_columns:
                dtype = str(df[col].dropna().dtype)
                col_type = dtype_map.get(dtype, String)
                if col_type == String:
                    max_len = df[col].dropna().astype(str).map(len).max() or 255
                    col_type = String(int(max_len))
                if isinstance(col_type, type):
                    col_type_instance = col_type()
                else:
                    col_type_instance = col_type
                alter_sql = f'ALTER TABLE "{schema}"."{table_name}" ADD COLUMN IF NOT EXISTS "{col}" {col_type_instance.compile(dialect=engine.dialect)}'
                with engine.begin() as conn:
                    conn.execute(text(alter_sql))
            print(f"[INFO] Columns added (if any were missing). Reloading table metadata.")
            metadata = MetaData(schema=schema)
            table = Table(table_name, metadata, autoload_with=engine)
            db_columns = {col.name: col for col in table.columns}
        # Optionally alter columns
        if alter_cols:
            print(f"[INFO] Checking for column length/type corrections...")
            for col, db_col in db_columns.items():
                if col in df.columns:
                    dtype = str(df[col].dropna().dtype)
                    if isinstance(db_col.type, String):
                        max_len = df[col].dropna().astype(str).map(len).max() or 1
                        if db_col.type.length is not None and max_len > db_col.type.length:
                            alter_sql = f'ALTER TABLE "{schema}"."{table_name}" ALTER COLUMN "{col}" TYPE VARCHAR({max_len})'
                            with engine.begin() as conn:
                                conn.execute(text(alter_sql))
            print(f"[INFO] Column length/type corrections complete.")
        # Only use columns present in DB
        valid_columns = set(table.columns.keys())
        use_columns = [col for col in df.columns if col in valid_columns]
        df = df[use_columns]
        unique_cols = set()
        pk_cols = set()
        if key_column is None:
            print(f"[INFO] key_column is None. Only insert operations are allowed. Forcing mode to 'insert'.")
            mode = "insert"
        else:
            inspector = inspect(engine)
            unique_constraints = inspector.get_unique_constraints(table_name, schema=schema)
            pk_constraint = inspector.get_pk_constraint(table_name, schema=schema)
            for uc in unique_constraints:
                unique_cols.update(uc['column_names'])
            pk_cols = set(pk_constraint.get('constrained_columns', []))
        # Helper: get existing keys
        def get_existing_keys():
            with engine.connect() as conn:
                result = conn.execute(text(f'SELECT "{key_column}" FROM "{schema}"."{table_name}"'))
                return set(row[0] for row in result)
        if mode == "insert" or (mode == "upsert" and (key_column is None or (key_column not in unique_cols and key_column not in pk_cols))):
            if mode == "upsert" and key_column is not None:
                print(f"[WARN] '{key_column}' is not a unique or primary key in {schema}.{table_name}. Falling back to insert-only for new rows.")
            if key_column is not None:
                existing_keys = get_existing_keys()
                df_new = df[~df[key_column].isin(existing_keys)]
            else:
                df_new = df
            if not df_new.empty:
                print(f"[INFO] Inserting {len(df_new)} new rows into {schema}.{table_name}.")
                data = df_new.to_dict(orient='records')
                stmt = insert(table).values(data)
                with engine.begin() as conn:
                    conn.execute(stmt)
            else:
                print(f"[INFO] No new rows to insert for {schema}.{table_name}.")
            print(f"[INFO] Insert operation complete.")
            return True
        elif mode == "update":
            if key_column is None:
                print(f"[ERROR] key_column must be provided for update mode. Aborting.")
                return False
            existing_keys = get_existing_keys()
            df_existing = df[df[key_column].isin(existing_keys)]
            if not df_existing.empty:
                print(f"[INFO] Updating {len(df_existing)} existing rows in {schema}.{table_name}.")
                with engine.begin() as conn:
                    for _, row in df_existing.iterrows():
                        update_dict = row.to_dict()
                        if key_column in update_dict:
                            update_dict = {k: v for k, v in update_dict.items() if k != key_column}
                        stmt = (
                            update(table)
                            .where(getattr(table.c, key_column) == row[key_column])
                            .values(**update_dict)
                        )
                        conn.execute(stmt)
            else:
                print(f"[INFO] No existing rows to update for {schema}.{table_name}.")
            print(f"[INFO] Update operation complete.")
            return True
        # Default: upsert (key_column is unique or PK)
        if key_column is not None:
            print(f"[INFO] Performing upsert for {len(df)} rows in {schema}.{table_name}.")
            with engine.begin() as conn:
                for _, row in df.iterrows():
                    row_dict = row.to_dict()
                    stmt = insert(table).values(**row_dict)
                    update_dict = row.to_dict()
                    if key_column in update_dict:
                        update_dict = {k: v for k, v in update_dict.items() if k != key_column}
                    stmt = stmt.on_conflict_do_update(
                        index_elements=[key_column],
                        set_=update_dict
                    )
                    conn.execute(stmt)
            print(f"[INFO] Upsert operation complete.")
            return True
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False



# initialize logging
logger = LoggerConfigurator('stock_analyzer',
                            r'C:\Users\dhaan\OneDrive\Documents\Python\AI Project\logs',
                            'DEBUG').configure()


