import pandas as pd
import hashlib
import os
import json
import jwt
import datetime
import logging
from typing import Optional, Dict, Any
from pymongo import MongoClient
import psycopg2
from psycopg2 import pool
from uuid import uuid4
from rag_simulation.schema import Query

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

 
class SQLDatabase:
    """Main database interface for PostgreSQL operations"""
    def __init__(self, db_name: str):
        """
        Initializes the SQLDatabase instance with the given database name.

        Args:
            db_name (str): The name of the PostgreSQL database.
        """
        self.con = None
        self.cursor = None
        self.db_name = db_name
        self.SECRET_KEY = 42
        # self.SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'your-secret-key')
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
 
                host=os.getenv('POSTGRES_HOST', "localhost"),  # Use the correct service name
                port=int(os.getenv('POSTGRES_PORT', "32003")) # 32003
                options="-c client_encoding=UTF8"

            )
            self.cursor = self.con.cursor()
            self.check_table_existence()
            logger.info("Connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def check_table_existence(self):
        """Check if required tables exist, create if not"""
        try:
            self.cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = %s
                )
            """, ('users',))
            if not self.cursor.fetchone()[0]:
                self.create_tables()
        except Exception as e:
            logger.error(f"Table check failed: {e}")
            raise

    def create_tables(self):
        """Create required tables from SQL file"""
        try:
            with open('sql/init.sql', 'r') as file:
                self.cursor.execute(file.read())
            self.con.commit()
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Table creation failed: {e}")
            raise

    def verifier_si_utilisateur(self, username: str) -> bool:
        """
        Vérifie si un utilisateur existe déjà.
        
        Args:
            username (str): Nom d'utilisateur à vérifier
            
        Returns:
            bool: True si l'utilisateur existe, False sinon
        """
        try:
            self.cursor.execute("""
                SELECT EXISTS (
                    SELECT 1 
                    FROM users 
                    WHERE username = %s
                )
            """, (username,))
            return self.cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de l'utilisateur: {e}")
            raise

    def create_user(self, username: str, password: str, role: str = 'user'):
        """Create new user with secure password hash"""
        password_hash = hashlib.md5(password.encode()).hexdigest()
        # user_id = str(uuid4())
        try:
            self.cursor.execute("""
                INSERT INTO users (username, password_hash,  role) 
                VALUES (%s, %s, %s)
                RETURNING username
            """, ( username, password_hash,  role))
            self.con.commit()
            return self.cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"User creation failed: {e}")
            raise

    def verify_password(self, username: str, password: str) -> bool:
        """Verify password with MD5 hash"""
        try:
            self.cursor.execute("""
                SELECT password_hash 
                FROM users 
                WHERE username = %s
            """, (username,))
            result = self.cursor.fetchone()
            if result:
                stored_hash = result[0]
                password_hash = hashlib.md5(password.encode()).hexdigest()
                return password_hash == stored_hash
            return False
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            raise

    def login_user(self, username: str, password: str, role: str) -> Optional[str]:
        """Login user and return JWT token"""
        if self.verify_password(username, password):
            return self.generate_jwt_token(username, role)
        return None


    def create_chat_session(self, username: str, title: str) -> str:
        """Create new chat session"""
        try:
            self.cursor.execute("""
                INSERT INTO chat_sessions (chat_title, username)
                VALUES (%s, %s)
                RETURNING chat_title
            """, (title, username))
            self.con.commit()
            return self.cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Chat session creation failed: {e}")
            raise




    # def get_metrics(self) -> Dict[str, float]:
    #     """Get system metrics from database"""
    #     query = """
    #     SELECT 
    #         AVG(latency) as avg_latency,
    #         COUNT(DISTINCT username) as total_users,
    #         COUNT(DISTINCT chat_title) as active_chats,
    #         AVG(CASE WHEN safe THEN 1 ELSE 0 END) * 100 as success_rate
    #     FROM chatbot_history
    #     WHERE created_at >= NOW() - INTERVAL '24 hours'
    #     """
    #     return self.execute_query(query)[0]

    # def get_hourly_stats(self) -> List[Dict]:
    #     """Get hourly statistics"""
    #     query = """
    #     SELECT 
    #         EXTRACT(YEAR FROM timestamp) as year,
    #         EXTRACT(MONTH FROM timestamp) as month,
    #         EXTRACT(DAY FROM timestamp) as day,
    #         EXTRACT(HOUR FROM timestamp) as hour,
    #         AVG(latency) as avg_latency,
    #         COUNT(*) as query_count
    #     FROM chatbot_history
    #     GROUP BY year, month, day, hour
    #     ORDER BY year, month, day, hour
    #     """
    #     return self.execute_query(query)

    def ask_line_plot_user(self, username: str) -> pd.DataFrame:
        """
        Generates and displays a line plot of average latency per hour.

        Retrieves data from the `chatbot_history` table, computes hourly averages,
        and visualizes it using a Streamlit line chart.

        Returns: 
            None
        """
        self.cursor.execute("""
        SELECT 
            timestamp,
            latency as avg_latency
            
        FROM chatbot_history
        WHERE username = %s
        """, (username,))
        df_line_plot = pd.DataFrame(self.cursor.fetchall(), columns=['timestamp', 'avg_latency'])
        df_line_plot["month"] = pd.to_datetime(df_line_plot["timestamp"]).dt.month
        df_line_plot["day"] = pd.to_datetime(df_line_plot["timestamp"]).dt.day
        df_line_plot["hour"] = pd.to_datetime(df_line_plot["timestamp"]).dt.hour
        df_line_plot["datetime"] = pd.to_datetime(df_line_plot["timestamp"])
        #group by year, month, day, hour
        df_line_plot = df_line_plot.groupby(["month", "day", "hour"]).mean().reset_index()


        # df_line_plot["datetime"] = pd.to_datetime(
        #     df_line_plot[["year", "month", "day", "hour"]]
        # )
        df_line_plot.set_index("datetime", inplace=True)
        print('df_line_plot')
        print(df_line_plot)
        return df_line_plot


    def  ask_line_plot(self) -> pd.DataFrame:
        """
        Generates and displays a line plot of average latency per hour.

        Retrieves data from the `chatbot_history` table, computes hourly averages,
        and visualizes it using a Streamlit line chart.

        Returns: 
            None
        """
        self.cursor.execute("""
        SELECT 
            EXTRACT(YEAR FROM timestamp) as year,
            EXTRACT(MONTH FROM timestamp) as month,
            EXTRACT(DAY FROM timestamp) as day,
            EXTRACT(HOUR FROM timestamp) as hour,
            AVG(latency) as avg_latency,
            COUNT(*) as query_count
        FROM chatbot_history
        GROUP BY year, month, day, hour
        ORDER BY year, month, day, hour
        """)
        df_line_plot = pd.DataFrame(self.cursor.fetchall(), columns=['year', 'month', 'day', 'hour', 'avg_latency'])
        df_line_plot["datetime"] = pd.to_datetime(
            df_line_plot[["year", "month", "day", "hour"]]
        )
        df_line_plot.set_index("datetime", inplace=True)
        return df_line_plot

    def add_query(self, query: Query, chat_title: str, username: str): 
        """
        Adds a query to the 'chatbot_history' table.
        
        Args:
            query (Query): Query object containing message details
            chat_title (str): Title of the chat session
            username (str): Username of the sender
        """
        self.check_table_existence()

        #check if chat_title exists
        self.cursor.execute("SELECT chat_title FROM chat_sessions WHERE chat_title = %s", (chat_title,))
        if not self.cursor.fetchone():
            self.create_chat_session(username, chat_title)
        
        insert_query = """
        INSERT INTO chatbot_history (
             chat_title, username, query, answer, 
            embedding_model, generative_model, context,
            safe, latency, completion_tokens, prompt_tokens,
            query_price, energy_usage, gwp
        ) VALUES (
             %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        """

        try:
            self.cursor.execute(
                insert_query,
                (
                    # query.query_id,
                    chat_title,
                    username,
                    query.query,
                    query.answer,
                    query.embedding_model,
                    query.generative_model,
                    str(query.context),
                    query.safe,
                    query.latency,
                    query.completion_tokens,
                    query.prompt_tokens,
                    query.query_price,
                    query.energy_usage,
                    query.gwp
                )
            )
            # self.con.commit()
            logger.info("Query added successfully")
        
        #update chat_sessions table  updated_at
            self.cursor.execute("""
                UPDATE chat_sessions
                SET updated_at = NOW()
                WHERE chat_title = %s
            """, (chat_title,))
            logger.info("Chat session updated successfully")
            self.con.commit()

        except Exception as e:
            logger.error(f"Error adding query: {e}")
            self.con.rollback()
            raise

    def get_user_role(self, username: str) -> str:
        """Retrieve user role from database"""
        try:
            self.cursor.execute("""
                SELECT role 
                FROM users 
                WHERE username = %s
            """, (username,))
            return self.cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Role retrieval failed: {e}")
            raise

    def get_users(self) -> list:
        """Retrieve all users from database"""
        try:
            self.cursor.execute("SELECT username FROM users")
            return [row[0] for row in self.cursor.fetchall()]
        except Exception as e:
            logger.error(f"User retrieval failed: {e}")
            raise
    
    def get_all_chats(self) -> list:
        """Retrieve all chat history from database"""
        try:
            self.cursor.execute("""
                SELECT chat_title, query, answer, created_at
                FROM chatbot_history
                ORDER BY created_at ASC
            """)
            return self.cursor.fetchall()
        except Exception as e:
            logger.error(f"Chat history retrieval failed: {e}")
            raise
    
    def get_chat_history(self, username: str) -> list:
        """Retrieve chat history with sessions"""
        try:
            self.cursor.execute("""
                SELECT 
                    ch.chat_title,
                    JSON_AGG(
                        JSON_BUILD_OBJECT(
                            'role', CASE WHEN ch.answer IS NULL THEN 'user' ELSE 'assistant' END,
                            'content', COALESCE(ch.answer, ch.query),
                            'timestamp', ch.created_at
                        ) ORDER BY ch.created_at
                    ) as messages,
                    MIN(ch.created_at) as date,
                    COUNT(*) as message_count
                FROM chatbot_history ch
                WHERE ch.username = %s
                GROUP BY ch.chat_title
                ORDER BY MIN(ch.created_at) DESC
            """, (username,))
            
            chats = []
            for row in self.cursor.fetchall():
                chats.append({
                    'title': row[0],
                    'messages': row[1] if row[1] else [],
                    'date': row[2],
                    'message_count': row[3],
                    'chat_id': row[0]  # Using chat_title as ID
                })
            return chats
        
        except Exception as e:
            logger.error(f"Chat history retrieval failed: {e}")
            raise
 

    # def get_chat_history(self,  username: str) -> list:
    #     """Retrieve chat history"""
    #     try:
    #         self.cursor.execute("""
    #             SELECT chat_title, query, answer, created_at
    #             FROM chatbot_history
    #             WHERE username = %s
    #             ORDER BY created_at ASC
    #         """, (username,))
    #         return self.cursor.fetchall()
    #     except Exception as e:
    #         logger.error(f"Chat history retrieval failed: {e}")
    #         raise
    
    def get_chat_sessions(self, username: str) -> list:
        """Retrieve chat sessions"""
        try:
            self.cursor.execute("""
                SELECT chat_title, updated_at
                FROM chat_sessions
                WHERE username = %s
                ORDER BY updated_at DESC
            """, (username,))
            return self.cursor.fetchall()
        except Exception as e:
            logger.error(f"Chat session retrieval failed: {e}")
            raise


    # def get_chat_history(self, chat_title: str) -> list:
    #     """Retrieve chat history"""
    #     try:
    #         self.cursor.execute("""
    #             SELECT query, answer 
    #             FROM chatbot_history 
    #             WHERE chat_title = %s 
    #             ORDER BY created_at ASC
    #         """, (chat_title,))
    #         return self.cursor.fetchall()
    #     except Exception as e:
    #         logger.error(f"Chat history retrieval failed: {e}")
    #         raise

    def generate_jwt_token(self, user_id: str, role: str) -> str:
        """Generate JWT token"""
        try:
            payload = {
                'user_id': user_id,
                'role': role,
                'exp': datetime.datetime.utcnow() + datetime.timedelta(days=1)
            }
            return jwt.encode(payload, self.SECRET_KEY, algorithm='HS256')
        except Exception as e:
            logger.error(f"Token generation failed: {e}")
            raise

def hash_password(password: str) -> tuple[str, str]:
    """Hash password with salt"""
    # salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        # salt,
        100000
    )
    # return salt.hex(), key.hex()
    return  key.hex()

class MongoDB:
    """MongoDB management for document storage"""
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
        self.client = None
        self.db = None
        self.collection = None
        self.db_name = db_name
        self.collection_name = collection_name
        self.data_dir = data_dir
        self.host = host
        self.port = port
        self.connect_to_mongodb()

    def connect_to_mongodb(self):
        """
        Connects to the MongoDB database.
        """
        try:
            self.client = MongoClient(self.host, self.port)
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            if self.collection.estimated_document_count() == 0:
                self.setup_database()
            print("Connected to MongoDB database")
        except Exception as e:
            print(f"Failed to connect to MongoDB: {e}")
            # raise

    def create_collection(self, db_name: str, collection_name: str):
        """Create a collection in the specified database"""
        db = self.client[db_name]
        collection = db[collection_name]
        return collection
    
    def insert_item(self, db_name: str, collection_name: str, item: dict):
        """Insert an item into the specified collection"""
        collection = self.create_collection(db_name, collection_name)
        collection.update_one(
            {'id': item['id'], 'ville': item['ville']},
            {'$set': item},
            upsert=True
        )
    
    def query_collection(self, db_name: str, collection_name: str, query: dict):
        """Query the specified collection"""
        collection = self.create_collection(db_name, collection_name)
        results = collection.find(query)
        return list(results)
    
    def setup_database(self):
        """Setup the database by inserting data from all JSON files in the specified directory"""
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


# Initialize database
db = SQLDatabase(db_name="poc_rag")