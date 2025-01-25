import pandas as pd
from rag_simulation.schema import Query

import os
import json
from pymongo import MongoClient
import psycopg2


class MongoDB:
    def __init__(self, db_name: str, collection_name: str, data_dir: str = 'data', host: str = 'localhost', port: int = 27017):
        """
        Initialize the MongoDB client with the specified host and port.
        
        Args:
            db_name (str): Name of the database
            collection_name (str): Name of the collection
            data_dir (str): Directory containing JSON files
            host (str): MongoDB host address
            port (int): MongoDB port number
        """
        self.client = MongoClient(host, port)
        self.db_name = db_name
        self.collection_name = collection_name
        self.data_dir = data_dir
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        
        # Check if the collection is empty and load data if necessary
        if self.collection.estimated_document_count() == 0:
            self.setup_database()

    def create_collection(self, db_name: str, collection_name: str):
        """
        Create a collection in the specified database.
        
        Args:
            db_name (str): Name of the database
            collection_name (str): Name of the collection
            
        Returns:
            Collection: The created collection
        """
        db = self.client[db_name]
        collection = db[collection_name]
        return collection
    
    def insert_item(self, db_name: str, collection_name: str, item: dict):
        """
        Insert an item into the specified collection.
        
        Args:
            db_name (str): Name of the database
            collection_name (str): Name of the collection
            item (dict): Item to insert
        """
        collection = self.create_collection(db_name, collection_name)
        collection.update_one(
            {'id': item['id'], 'ville': item['ville']},  # Filter to check if the item already exists
            {'$set': item},      # Update the item if it exists
            upsert=True          # Insert the item if it doesn't exist
        )
        # print("Inserted item with ID:", item['id'])
    
    def query_collection(self, db_name: str, collection_name: str, query: dict):
        """
        Query the specified collection.
        
        Args:
            db_name (str): Name of the database
            collection_name (str): Name of the collection
            query (dict): Query to execute
            
        Returns:
            list: List of query results
        """
        collection = self.create_collection(db_name, collection_name)
        results = collection.find(query)
        return list(results)
    
    def setup_database(self):
        """
        Setup the database by inserting data from all JSON files in the specified directory.
        """
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith('.json'):
                file_path = os.path.join(self.data_dir, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for item in data:
                    mandatory = ['id', 'ville', 'type_dechet', 'produits', 'action', 'instructions']
                    for field in mandatory:
                        if field not in item:
                            raise ValueError(f"Missing field '{field}' in item with ID: {item['id']}")
                    self.insert_item(self.db_name, self.collection_name, item)
        
        # print(f"Data from all JSON files in '{self.data_dir}' verified and inserted into MongoDB collection '{self.collection_name}' if not already present")


class SQLDatabase:
    def __init__(self, db_name: str):
        """
        Initializes the SQLDatabase instance with the given database name.

        Args:
            db_name (str): The name of the PostgreSQL database.
        """
        self.con = None
        self.cursor = None
        self.db_name = db_name
        self.connect_to_postgresql()

    def connect_to_postgresql(self):
        """
        Connects to the PostgreSQL database.
        """
        try:
            self.con = psycopg2.connect(
                dbname=os.getenv('POSTGRES_DBNAME', 'llm'),
                user=os.getenv('POSTGRES_USER', 'llm'),
                password=os.getenv('POSTGRES_PASSWORD', 'llm'),
                host='localhost',  # Use the correct service name
                port=os.getenv('POSTGRES_PORT', 5432)  # Use the correct port
            )
            self.cursor = self.con.cursor()
            print("Connected to PostgreSQL database")
        except Exception as e:
            pass
            print(f"Failed to connect to PostgreSQL: {e}")
            # raise

    def check_table_existence(self) -> None:
        """
        Checks if the 'chatbot_history' table exists in the database. If it doesn't, it creates it.
        """
        try:
            self.cursor.execute(
                """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema='public' AND table_name=%s;
                """,
                ("chatbot_history",),
            )
            table_exists = self.cursor.fetchone()
            if not table_exists:
                self.create_table()
        except psycopg2.Error as e:
            print(f"Error checking table existence: {e}")
            self.con.rollback()
            raise

    def create_table(self) -> None:
        """
        Creates the 'chatbot_history' table in the database with predefined columns.
        """
        with open('/Users/halcolo/Documents/code/lyon2/MASTER 2/LLM/RePilot-recycle-chatbot/sql/init.sql', 'r') as file:
            create_table_query = file.read()
        self.cursor.execute(create_table_query)
        self.con.commit()
        print("Table 'chatbot_history' created successfully.")

    def add_query(self, query: Query):
        """
        Adds a query to the 'chatbot_history' table.

        Args:
            query (Query): An instance of the Query class containing query details.
        """
        self.check_table_existence()

        # Extract year, month, day, hour, and minute from the timestamp
        year = query.timestamp.year
        month = query.timestamp.month
        day = query.timestamp.day
        hour = query.timestamp.hour
        minutes = query.timestamp.minute

        # Insert the query into the database
        insert_query = """
        INSERT INTO chatbot_history (
            query_id, query, answer, embedding_model, generative_model, context, 
            year, month, day, hour, minutes, safe, latency, 
            completion_tokens, prompt_tokens, query_price,
            energy_usage, gwp
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        self.cursor.execute(
            insert_query,
            (
                query.query_id,
                query.query,
                query.answer,
                query.embedding_model,
                query.generative_model,
                str(query.context),  # Ensure context is a string
                year,
                month,
                day,
                hour,
                minutes,
                query.safe,
                query.latency,
                query.completion_tokens,
                query.prompt_tokens,
                query.query_price,
                query.energy_usage,
                query.gwp,
            ),
        )
        self.con.commit()
        print("Query added successfully.")

    def ask_db(self, sql_query: str) -> pd.DataFrame:
        """
        Executes a SQL query and returns the result as a DataFrame.

        Args:
            sql_query (str): The SQL query to be executed.

        Returns:
            pd.DataFrame: The result of the query.
        """
        df_query: pd.DataFrame = pd.read_sql_query(sql_query, self.con)
        return df_query


db = SQLDatabase(db_name="poc_rag")
