from typing import List
import pandas as pd
import hashlib
import os
import json
import logging
from pymongo import MongoClient
import psycopg2
from rag_simulation.schema import Query
import time
import litellm
import os
import functools
import streamlit as st


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=100)  # 🔹 Stocke jusqu'à 100 réponses générées
def cached_generate_fake_answers(question: str, correct_answer: str):
    """
    Vérifie si la question a déjà été traitée récemment.
    Si oui, récupère la réponse stockée sans refaire appel à l'API.
    Sinon, génère de nouvelles réponses incorrectes.

    Args:
        question (str): La question du quiz.
        correct_answer (str): La réponse correcte.

    Returns:
        tuple[str, list[str]]: (Bonne réponse courte, [fausse réponse 1, fausse réponse 2])
    """
    return SQLDatabase.generate_fake_answers(question, correct_answer)


@st.cache_resource
def get_db_connection():
    """Établit une connexion persistante avec PostgreSQL."""
    return psycopg2.connect(
        dbname=os.getenv('POSTGRES_DBNAME', 'llm'),
        user=os.getenv('POSTGRES_USER', 'llm'),
        password=os.getenv('POSTGRES_PASSWORD', 'llm'),
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=os.getenv('POSTGRES_PORT', 25130)
    )


class MongoDB:
    def __init__(
        self, 
        db_name: str, 
        collection_name: str, 
        data_dir: str = 'data', 
        host: str = 'localhost', 
        port: int = 27017, 
        uri: str = None):    
        """
        Initialize the MongoDB client with the specified host and port.

        Args:
            db_name (str): Name of the database
            collection_name (str): Name of the collection
            data_dir (str): Directory containing JSON files
            host (str): MongoDB host address
            port (int): MongoDB port number
            uri (str): MongoDB connection URI
        """
        self.db_name = db_name
        self.collection = None
        self.collection_name = collection_name
        self.data_dir = data_dir
        self.host = host
        self.port = int(port)
        self.uri = uri
        self.connect_to_mongodb()
        
        if self.host is not None:
            self.client = MongoClient(self.host, self.port)
        elif self.uri is not None:
            self.client = MongoClient(self.uri)
        else:
            raise ValueError("Please provide either a host address or a connection URI")
        self.db = None

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
            {"id": item["id"], "ville": item["ville"]}, {"$set": item}, upsert=True
        )

    def query_collection(self, db_name: str, collection_name: str, query: dict):
        """Query the specified collection"""
        collection = self.create_collection(db_name, collection_name)
        results = collection.find(query)
        return list(results)

    def setup_database(self):
        """Setup the database by inserting data from all JSON files in the specified directory"""
        for file_name in os.listdir(self.data_dir):
            if file_name.endswith(".json"):
                file_path = os.path.join(self.data_dir, file_name)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for item in data:
                    mandatory = [
                        "id",
                        "ville",
                        "type_dechet",
                        "produits",
                        "action",
                        "instructions",
                    ]
                    for field in mandatory:
                        if field not in item:
                            raise ValueError(
                                f"Missing field '{field}' in item with ID: {item['id']}"
                            )
                    self.insert_item(self.db_name, self.collection_name, item)


class SQLDatabase:
    def __init__(self, db_name: str):
        """
        Initialise la connexion à la base de données PostgreSQL.

        Args:
            db_name (str): Nom de la base de données.
        """
        self.con = get_db_connection()
        self.cursor = self.con.cursor()
        self.db_name = db_name
        self.initialize_database()
        self.check_table_existence()

    def initialize_database(self):
        """Vérifie et initialise les tables si elles n'existent pas."""
        try:
            self.cursor.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='public';"
            )
            existing_tables = {table[0] for table in self.cursor.fetchall()}
            required_tables = {
                "chatbot_history",
                "chatbot_feedback",
                "quiz_questions",
                "quiz_responses",
                "users",
            }

            missing_tables = required_tables - existing_tables
            if missing_tables:
                print(f"⚠️ Création des tables manquantes : {missing_tables}")
                with open("sql/init.sql", "r") as file:
                    sql_script = file.read()
                    self.cursor.execute(sql_script)
                    self.con.commit()
                    print("✅ Base de données initialisée avec succès !")
        except Exception as e:
            print(f"❌ Erreur lors de l'initialisation de la base de données : {e}")
            self.con.rollback()

    def add_query(
        self,
        query_id: str,
        query: str,
        answer: str,
        embedding_model: str,
        generative_model: str,
        context: str,
        safe: bool,
        latency: float,
        completion_tokens: int,
        prompt_tokens: int,
        query_price: float,
        energy_usage: float,
        gwp: float,
        username: str = "user",
    ):
        """
        Ajoute une question et une réponse dans l'historique des conversations.
        """
        insert_query = """
        INSERT INTO chatbot_history (
            query_id, query, answer, embedding_model, generative_model, context, 
            safe, latency, completion_tokens, prompt_tokens, query_price,
            energy_usage, gwp, username, timestamp
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        """
        try:
            self.cursor.execute(
                insert_query,
                (
                    query_id,
                    query,
                    answer,
                    embedding_model,
                    generative_model,
                    context,
                    safe,
                    latency,
                    completion_tokens,
                    prompt_tokens,
                    query_price,
                    energy_usage,
                    gwp,
                    username,
                ),
            )
            self.con.commit()

            print("✅ Query ajoutée avec succès.")
        except Exception as e:
            print(f"❌ Erreur lors de l'ajout de la requête : {e}")
            self.con.rollback()

    def save_feedback(
            self, query_id: str, username: str, feedback: str, comment: str = None
        ) -> bool:
        """Enregistre le feedback de l'utilisateur."""
        try:
            insert_query = """
            INSERT INTO chatbot_feedback (query_id, username, feedback, comment, timestamp)
            VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
            """
            self.cursor.execute(insert_query, (query_id, username, feedback, comment))
            self.con.commit()
            print(f"✅ Feedback ajouté pour {query_id} : {feedback}")
            return True
        except Exception as e:
            print(f"❌ Erreur lors de l'ajout du feedback : {e}")
            self.con.rollback()
            return False

    @staticmethod
    def generate_fake_answers(
        question: str, correct_answer: str, max_retries: int = 3
    ) -> tuple[str, list[str]]:
        """
        Génère une version courte de la bonne réponse et deux fausses réponses via Mistral,
        en gérant les erreurs de rate limit avec un backoff.

        Args:
            question (str): La question de base.
            correct_answer (str): La réponse correcte détaillée.
            max_retries (int): Nombre maximal de tentatives en cas d'erreur.

        Returns:
            tuple[str, list[str]]: (Bonne réponse courte, [fausse réponse 1, fausse réponse 2])
        """
        attempt = 0
        while attempt < max_retries:
            try:
                # 🔹 Étape 1 : Reformuler la bonne réponse en version courte
                reformulation_prompt = f"""
                Voici une réponse détaillée : 
                "{correct_answer}"

                📌 Reformule-la en **une phrase courte et claire** qui garde son sens.
                🔹 Pas plus d'une ligne.
                🔹 Ne change pas le sens.

                Réponds seulement avec la réponse reformulée.
                """
                short_correct_answer = litellm.completion(
                    model="mistral/mistral-large-latest",
                    messages=[{"role": "user", "content": reformulation_prompt}],
                    max_tokens=50,
                    temperature=1.0,
                    api_key=os.getenv("MISTRAL_API_KEY"),
                )["choices"][0]["message"]["content"].strip()

                # 🔹 Étape 2 : Générer les mauvaises réponses
                prompt_fake_answers = f"""
                Tu es un générateur de quiz. Donne **exactement** 2 fausses réponses distinctes et réalistes.

                📌 **Règles à suivre** :
                - **N'ajoute pas** de phrases d'introduction.
                - **Tu dois inclure le nom de la ville mentionnée dans la question.**  
                - **Ne répète pas** la question.
                - Écris **directement** les 2 mauvaises réponses, **chacune sur une ligne**.

                ❓ **Question** : "{question}"
                ✅ **Bonne réponse** : "{correct_answer}"

                🔹 **Mauvaises réponses** :
                1. 
                2. 

                """
                response_fake = (
                    litellm.completion(
                        model="mistral/mistral-large-latest",
                        messages=[{"role": "user", "content": prompt_fake_answers}],
                        max_tokens=100,
                        temperature=1.2,
                        api_key=os.getenv("MISTRAL_API_KEY"),
                    )["choices"][0]["message"]["content"]
                    .strip()
                    .split("\n")
                )

                # ✅ Nettoyer et extraire les réponses
                fake_answers = [ans.strip("- ") for ans in response_fake if ans.strip()]
                fake_answers = (
                    fake_answers[:2]
                    if len(fake_answers) >= 2
                    else ["Réponse incorrecte 1", "Réponse incorrecte 2"]
                )

                return short_correct_answer, fake_answers

            except litellm.RateLimitError:
                attempt += 1
                wait_time = 2**attempt  # ⏳ Exponential Backoff
                print(
                    f"⚠️ Rate limit atteint. Nouvelle tentative dans {wait_time} secondes..."
                )
                time.sleep(wait_time)

            except Exception as e:
                print(f"❌ Erreur lors de la génération des réponses : {e}")
                return correct_answer, ["Réponse incorrecte 1", "Réponse incorrecte 2"]

        print("❌ Échec après plusieurs tentatives. Retour à une valeur par défaut.")
        return correct_answer, ["Réponse incorrecte 1", "Réponse incorrecte 2"]

    def get_quiz_questions(self, username: str, limit: int = 5) -> list[dict]:
        """
        Récupère un quiz basé sur les questions posées par l'utilisateur et génère des réponses incorrectes via le LLM.

        Args:
            username (str): Nom de l'utilisateur.
            limit (int, optional): Nombre de questions à récupérer. Defaults to 5.

        Returns:
            list[dict]: Liste de questions avec "question", "correct_answer", et "fake_answers".
        """
        try:
            # ✅ Vérifier si l'utilisateur existe
            self.cursor.execute(
                "SELECT COUNT(*) FROM users WHERE username = %s;", (username,)
            )
            user_exists = self.cursor.fetchone()[0]

            if user_exists == 0:
                print(f"⚠️ L'utilisateur {username} n'existe pas dans la table users.")
                return [{"message": "⚠️ Aucun compte trouvé pour cet utilisateur."}]

            # ✅ Récupérer les questions aléatoires
            self.cursor.execute(
                """
                SELECT query, answer 
                FROM chatbot_history
                WHERE username = %s  
                ORDER BY RANDOM()
                LIMIT %s;
                """,
                (username, limit),
            )
            questions = self.cursor.fetchall()

            if not questions:
                return [
                    {
                        "message": "⚠️ Aucun historique de questions trouvé pour cet utilisateur."
                    }
                ]

            quiz_data = []
            for query, correct_answer in questions:
                # 🔥 Générer la version courte de la bonne réponse + 2 fausses réponses
                short_correct_answer, fake_answers = cached_generate_fake_answers(
                    query, correct_answer
                )

                quiz_data.append(
                    {
                        "question": query,
                        "correct_answer": short_correct_answer,
                        "fake_answers": fake_answers,
                    }
                )

            return quiz_data

        except Exception as e:
            self.con.rollback()  # ✅ Rétablir la base en cas d’erreur
            print(f"❌ Erreur lors de la récupération des questions du quiz : {e}")
            return []

    def check_table_existence(self):
        """Check if required tables exist, create if not"""
        try:
            self.cursor.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = %s
                )
            """,
                ("users",),
            )
            if not self.cursor.fetchone()[0]:
                self.create_tables()
        except Exception as e:
            logger.error(f"Table check failed: {e}")
            raise

    def create_tables(self):
        """Create required tables from SQL file"""
        try:
            with open("sql/init.sql", "r") as file:
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
            self.cursor.execute(
                """
                SELECT EXISTS (
                    SELECT 1 
                    FROM users 
                    WHERE username = %s
                )
            """,
                (username,),
            )
            return self.cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de l'utilisateur: {e}")
            raise

    def create_user(self, username: str, password: str, role: str = "user"):
        """Create new user with secure password hash"""
        password_hash = hashlib.md5(password.encode()).hexdigest()
        try:
            self.cursor.execute(
                """
                INSERT INTO users (username, password_hash,  role) 
                VALUES (%s, %s, %s)
                RETURNING username
            """,
                (username, password_hash, role),
            )
            self.con.commit()
            return self.cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"User creation failed: {e}")
            raise

    def verify_password(self, username: str, password: str) -> bool:
        """Verify password with MD5 hash"""
        try:
            self.cursor.execute(
                """
                SELECT password_hash 
                FROM users 
                WHERE username = %s
            """,
                (username,),
            )
            result = self.cursor.fetchone()
            if result:
                stored_hash = result[0]
                password_hash = hashlib.md5(password.encode()).hexdigest()
                return password_hash == stored_hash
            return False
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            raise

    def create_chat_session(self, username: str, title: str) -> str:
        """Create new chat session"""
        try:
            self.cursor.execute(
                """
                INSERT INTO chat_sessions (chat_title, username)
                VALUES (%s, %s)
                RETURNING chat_title
            """,
                (title, username),
            )
            self.con.commit()
            return self.cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Chat session creation failed: {e}")
            raise

    def ask_line_plot_user(self, username: str) -> pd.DataFrame:
        """
        Generates and displays a line plot of average latency per hour.

        Retrieves data from the `chatbot_history` table, computes hourly averages,
        and visualizes it using a Streamlit line chart.

        Returns:
            None
        """
        self.cursor.execute(
            """
        SELECT 
            timestamp,
            latency as avg_latency,
            safe as avg_safe,
            completion_tokens as avg_completion_tokens,
            prompt_tokens as avg_prompt_tokens,
            query_price as avg_query_price,
            energy_usage as avg_energy_usage,
            gwp as avg_gwp,
            count(*) as query_count
   
        FROM chatbot_history
        WHERE username = %s
        GROUP BY timestamp, avg_latency, avg_safe, avg_completion_tokens, avg_prompt_tokens, avg_query_price, avg_energy_usage, avg_gwp
        """,
            (username,),
        )
        df_line_plot = pd.DataFrame(
            self.cursor.fetchall(),
            columns=[
                "timestamp",
                "avg_latency",
                "avg_safe",
                "avg_completion_tokens",
                "avg_prompt_tokens",
                "avg_query_price",
                "avg_energy_usage",
                "avg_gwp",
                "query_count",
            ],
        )
        df_line_plot["month"] = pd.to_datetime(df_line_plot["timestamp"]).dt.month
        df_line_plot["day"] = pd.to_datetime(df_line_plot["timestamp"]).dt.day
        df_line_plot["hour"] = pd.to_datetime(df_line_plot["timestamp"]).dt.hour
        # df_line_plot["hour"] = pd.to_datetime(df_line_plot["timestamp"]).dt.hour
        # faire  un time  range  entre  de 3 entre  4-12 et  12-20  et  20-4 ou  chaque  plage  horaire  devient une colone
        time_range = [(4, 12), (12, 20), (20, 4)]
        df_line_plot["4h - 12h_latency"] = 0
        df_line_plot["12h - 20h_latency"] = 0
        df_line_plot["20h - 4h_latency"] = 0

        df_line_plot["4h - 12h_safe"] = 0
        df_line_plot["12h - 20h_safe"] = 0
        df_line_plot["20h - 4h_safe"] = 0

        df_line_plot["4h - 12h_completion_tokens"] = 0
        df_line_plot["12h - 20h_completion_tokens"] = 0
        df_line_plot["20h - 4h_completion_tokens"] = 0

        df_line_plot["4h - 12h_prompt_tokens"] = 0
        df_line_plot["12h - 20h_prompt_tokens"] = 0
        df_line_plot["20h - 4h_prompt_tokens"] = 0

        df_line_plot["4h - 12h_query_price"] = 0
        df_line_plot["12h - 20h_query_price"] = 0
        df_line_plot["20h - 4h_query_price"] = 0

        df_line_plot["4h - 12h_energy_usage"] = 0
        df_line_plot["12h - 20h_energy_usage"] = 0
        df_line_plot["20h - 4h_energy_usage"] = 0

        df_line_plot["4h - 12h_gwp"] = 0
        df_line_plot["12h - 20h_gwp"] = 0
        df_line_plot["20h - 4h_gwp"] = 0

        df_line_plot["4h - 12h_query_count"] = 0
        df_line_plot["12h - 20h_query_count"] = 0
        df_line_plot["20h - 4h_query_count"] = 0

        for i, (start, end) in enumerate(time_range):
            df_line_plot.loc[
                (df_line_plot["hour"] >= start) & (df_line_plot["hour"] < end),
                f"{start}h - {end}h_latency",
            ] = df_line_plot["avg_latency"]
            df_line_plot.loc[
                (df_line_plot["hour"] >= start) & (df_line_plot["hour"] < end),
                f"{start}h - {end}h_safe",
            ] = df_line_plot["avg_safe"]
            df_line_plot.loc[
                (df_line_plot["hour"] >= start) & (df_line_plot["hour"] < end),
                f"{start}h - {end}h_completion_tokens",
            ] = df_line_plot["avg_completion_tokens"]
            df_line_plot.loc[
                (df_line_plot["hour"] >= start) & (df_line_plot["hour"] < end),
                f"{start}h - {end}h_prompt_tokens",
            ] = df_line_plot["avg_prompt_tokens"]
            df_line_plot.loc[
                (df_line_plot["hour"] >= start) & (df_line_plot["hour"] < end),
                f"{start}h - {end}h_query_price",
            ] = df_line_plot["avg_query_price"]
            df_line_plot.loc[
                (df_line_plot["hour"] >= start) & (df_line_plot["hour"] < end),
                f"{start}h - {end}h_energy_usage",
            ] = df_line_plot["avg_energy_usage"]
            df_line_plot.loc[
                (df_line_plot["hour"] >= start) & (df_line_plot["hour"] < end),
                f"{start}h - {end}h_gwp",
            ] = df_line_plot["avg_gwp"]
            df_line_plot.loc[
                (df_line_plot["hour"] >= start) & (df_line_plot["hour"] < end),
                f"{start}h - {end}h_query_count",
            ] = df_line_plot["query_count"]

        df_line_plot["datetime"] = pd.to_datetime(df_line_plot["timestamp"])
        # group by year, month, day, hour
        df_line_plot = df_line_plot.groupby(["datetime"]).mean().reset_index()

        df_line_plot.set_index("datetime", inplace=True)
        # print('df_line_plot')
        # print(df_line_plot)
        return df_line_plot

    def ask_line_plot(self) -> pd.DataFrame:
        """
        Generates and displays a line plot of average latency per hour.

        Retrieves data from the `chatbot_history` table, computes hourly averages,
        and visualizes it using a Streamlit line chart.

        Returns:
            None
        """
        self.cursor.execute(
            """
        SELECT 
            username,
            timestamp,
            latency as avg_latency,
            safe as avg_safe,
            completion_tokens as avg_completion_tokens,
            prompt_tokens as avg_prompt_tokens,
            query_price as avg_query_price,
            energy_usage as avg_energy_usage,
            gwp as avg_gwp,
            count(*) as query_count
   
        FROM chatbot_history
        
        GROUP BY  username, timestamp, avg_latency, avg_safe, avg_completion_tokens, avg_prompt_tokens, avg_query_price, avg_energy_usage, avg_gwp
        """
        )
        df_line_plot = pd.DataFrame(
            self.cursor.fetchall(),
            columns=[
                "username",
                "timestamp",
                "avg_latency",
                "avg_safe",
                "avg_completion_tokens",
                "avg_prompt_tokens",
                "avg_query_price",
                "avg_energy_usage",
                "avg_gwp",
                "query_count",
            ],
        )
        df_line_plot["month"] = pd.to_datetime(df_line_plot["timestamp"]).dt.month
        df_line_plot["day"] = pd.to_datetime(df_line_plot["timestamp"]).dt.day
        df_line_plot["hour"] = pd.to_datetime(df_line_plot["timestamp"]).dt.hour
        # df_line_plot["hour"] = pd.to_datetime(df_line_plot["timestamp"]).dt.hour
        # faire  un time  range  entre  de 3 entre  4-12 et  12-20  et  20-4 ou  chaque  plage  horaire  devient une colone
        time_range = [(4, 12), (12, 20), (20, 4)]
        df_line_plot["4h - 12h_latency"] = 0
        df_line_plot["12h - 20h_latency"] = 0
        df_line_plot["20h - 4h_latency"] = 0

        df_line_plot["4h - 12h_safe"] = 0
        df_line_plot["12h - 20h_safe"] = 0
        df_line_plot["20h - 4h_safe"] = 0

        df_line_plot["4h - 12h_completion_tokens"] = 0
        df_line_plot["12h - 20h_completion_tokens"] = 0
        df_line_plot["20h - 4h_completion_tokens"] = 0

        df_line_plot["4h - 12h_prompt_tokens"] = 0
        df_line_plot["12h - 20h_prompt_tokens"] = 0
        df_line_plot["20h - 4h_prompt_tokens"] = 0

        df_line_plot["4h - 12h_query_price"] = 0
        df_line_plot["12h - 20h_query_price"] = 0
        df_line_plot["20h - 4h_query_price"] = 0

        df_line_plot["4h - 12h_energy_usage"] = 0
        df_line_plot["12h - 20h_energy_usage"] = 0
        df_line_plot["20h - 4h_energy_usage"] = 0

        df_line_plot["4h - 12h_gwp"] = 0
        df_line_plot["12h - 20h_gwp"] = 0
        df_line_plot["20h - 4h_gwp"] = 0

        df_line_plot["4h - 12h_query_count"] = 0
        df_line_plot["12h - 20h_query_count"] = 0
        df_line_plot["20h - 4h_query_count"] = 0

        for i, (start, end) in enumerate(time_range):
            df_line_plot.loc[
                (df_line_plot["hour"] >= start) & (df_line_plot["hour"] < end),
                f"{start}h - {end}h_latency",
            ] = df_line_plot["avg_latency"]
            df_line_plot.loc[
                (df_line_plot["hour"] >= start) & (df_line_plot["hour"] < end),
                f"{start}h - {end}h_safe",
            ] = df_line_plot["avg_safe"]
            df_line_plot.loc[
                (df_line_plot["hour"] >= start) & (df_line_plot["hour"] < end),
                f"{start}h - {end}h_completion_tokens",
            ] = df_line_plot["avg_completion_tokens"]
            df_line_plot.loc[
                (df_line_plot["hour"] >= start) & (df_line_plot["hour"] < end),
                f"{start}h - {end}h_prompt_tokens",
            ] = df_line_plot["avg_prompt_tokens"]
            df_line_plot.loc[
                (df_line_plot["hour"] >= start) & (df_line_plot["hour"] < end),
                f"{start}h - {end}h_query_price",
            ] = df_line_plot["avg_query_price"]
            df_line_plot.loc[
                (df_line_plot["hour"] >= start) & (df_line_plot["hour"] < end),
                f"{start}h - {end}h_energy_usage",
            ] = df_line_plot["avg_energy_usage"]
            df_line_plot.loc[
                (df_line_plot["hour"] >= start) & (df_line_plot["hour"] < end),
                f"{start}h - {end}h_gwp",
            ] = df_line_plot["avg_gwp"]
            df_line_plot.loc[
                (df_line_plot["hour"] >= start) & (df_line_plot["hour"] < end),
                f"{start}h - {end}h_query_count",
            ] = df_line_plot["query_count"]

        df_line_plot["datetime"] = pd.to_datetime(df_line_plot["timestamp"])
        # group by year, month, day, hour
        # df_line_plot = df_line_plot.groupby(["datetime"]).mean().reset_index()

        df_line_plot.set_index("datetime", inplace=True)
        print("df_line_plot")
        print(df_line_plot)
        return df_line_plot

    def get_user_role(self, username: str) -> str:
        """Retrieve user role from database"""
        try:
            self.cursor.execute(
                """
                SELECT role 
                FROM users 
                WHERE username = %s
            """,
                (username,),
            )
            return self.cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Role retrieval failed: {e}")
            raise

    def get_usernames(self) -> list:  # Ne retourne que le username ?
        """Retrieve all users from database"""
        try:
            self.cursor.execute("SELECT username FROM users")
            return [row[0] for row in self.cursor.fetchall()]
        except Exception as e:
            logger.error(f"User retrieval failed: {e}")
            raise

    def fetch_all_users(self) -> list:
        """Retrieve all user information from the database"""
        try:
            self.cursor.execute("SELECT * FROM users")  # Récupère toutes les colonnes
            columns = [
                desc[0] for desc in self.cursor.description
            ]  # Liste des noms de colonnes
            return [
                dict(zip(columns, row)) for row in self.cursor.fetchall()
            ]  # Convertit chaque ligne en dict
        except Exception as e:
            logger.error(f"Failed to fetch all users: {e}")
            raise

    def get_all_chats(self) -> list:
        """Retrieve all chat history from database"""
        try:
            self.cursor.execute(
                """
                SELECT chat_title, query, answer, created_at
                FROM chatbot_history
                ORDER BY created_at ASC
            """
            )
            return self.cursor.fetchall()
        except Exception as e:
            logger.error(f"Chat history retrieval failed: {e}")
            raise

    def get_chat_history(self, username: str) -> list:
        """Retrieve chat history with sessions"""
        try:
            self.cursor.execute(
                """
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
            """,
                (username,),
            )

            chats = []
            for row in self.cursor.fetchall():
                chats.append(
                    {
                        "title": row[0],
                        "messages": row[1] if row[1] else [],
                        "date": row[2],
                        "message_count": row[3],
                        "chat_id": row[0],  # Using chat_title as ID
                    }
                )
            return chats

        except Exception as e:
            logger.error(f"Chat history retrieval failed: {e}")
            raise

    def get_chat_history_user(self, username: str) -> list:
        try:
            self.cursor.execute(
                """
                SELECT  query, answer, created_at
                FROM chatbot_history
                WHERE username = %s
                ORDER BY created_at ASC
            """,
                (username,),
            )
            df = pd.DataFrame(
                self.cursor.fetchall(), columns=["query", "answer", "created_at"]
            )
            df = df.set_index("created_at")
            liste = []
            for i in range(len(df)):
                liste.append({"role": "user", "content": df["query"][i]})
                liste.append({"role": "assistant", "content": df["answer"][i]})
            return liste

        except Exception as e:
            logger.error(f"Chat history retrieval failed: {e}")
            raise

    def get_chat_sessions(self, username: str) -> list:
        """Retrieve chat sessions"""
        try:
            self.cursor.execute(
                """
                SELECT chat_title, updated_at
                FROM chat_sessions
                WHERE username = %s
                ORDER BY updated_at DESC
            """,
                (username,),
            )
            return self.cursor.fetchall()
        except Exception as e:
            logger.error(f"Chat session retrieval failed: {e}")
            raise

    # ============================= Gestion admins =========================================

    ##########OVERVIEW PAGE##############
    def get_usage_statistics(self) -> dict:
        """Retrieve usage statistics"""
        try:
            self.cursor.execute(
                """
                WITH model_counts AS (
                SELECT generative_model, COUNT(*) as model_count
                FROM chatbot_history
                GROUP BY generative_model
                ORDER BY model_count DESC
                LIMIT 1
                )
                SELECT 
                    COUNT(DISTINCT username) AS total_users,
                    COUNT(DISTINCT chat_title) AS total_chats,
                    COUNT(*) AS total_queries,
                    AVG(latency) AS avg_latency,
                    SUM(query_price) AS total_cost,
                    SUM(energy_usage) AS total_energy,
                    SUM(gwp) AS total_gwp,
                    ROUND(
                        (COUNT(CASE WHEN safe = true THEN 1 END)::float / 
                        COUNT(*)::float * 100)::numeric, 
                        2
                    ) AS safe_queries_percentage,
                    (SELECT generative_model FROM model_counts) as most_used_model
                FROM chatbot_history
            """
            )
            result = self.cursor.fetchone()
            return {
                "total_users": result[0],
                "total_chats": result[1],
                "total_queries": result[2],
                "avg_latency": result[3],
                "total_cost": result[4],
                "total_energy": result[5],
                "total_gwp": result[6],
                "safe_queries_percentage": result[7],
                "most_used_model": result[8],
            }
        except Exception as e:
            logger.error(f"Usage statistics retrieval failed: {e}")
            raise

    def get_activity_data(self, start_date: str, end_date: str):
        query = """
        SELECT 
            DATE(created_at) AS activity_date, 
            COUNT(DISTINCT username) AS user_count,
            COUNT(query_id) AS query_count
        FROM chatbot_history
        WHERE DATE(created_at) BETWEEN %s AND %s  -- Cast vers une date sans heures
        GROUP BY activity_date
        ORDER BY activity_date;
        """

        self.cursor.execute(query, (start_date, end_date))
        return self.cursor.fetchall()

    ##########USERS PAGE##############

    def delete_user_and_data(self, username: str) -> bool:
        """Delete a user and all associated data"""
        try:
            # Supprimer les données associées dans les autres tables
            self.cursor.execute(
                "DELETE FROM chatbot_history WHERE username = %s", (username,)
            )
            self.cursor.execute(
                "DELETE FROM chatbot_feedback WHERE username = %s", (username,)
            )
            self.cursor.execute(
                "DELETE FROM quiz_questions WHERE username = %s", (username,)
            )
            self.cursor.execute(
                "DELETE FROM quiz_responses WHERE username = %s", (username,)
            )
            self.cursor.execute(
                "DELETE FROM chat_sessions WHERE username = %s", (username,)
            )

            # Supprimer l'utilisateur
            self.cursor.execute("DELETE FROM users WHERE username = %s", (username,))
            self.con.commit()
            return True
        except Exception as e:
            logger.error(f"User deletion failed: {e}")
            self.con.rollback()
            raise

    def get_top_users_by_metric(self, metric: str) -> List[tuple]:
        """
        Retrieve the top 5 users by the selected metric (money, environmental impact, or latency).

        Args:
            metric (str): The metric to analyze ("Money Spent", "Environmental Impact", or "Latency").

        Returns:
            List[tuple]: A list of tuples containing (username, metric_value).
        """
        try:
            if metric == "Money Spent":
                query = """
                    SELECT username, SUM(query_price) AS total_cost
                    FROM chatbot_history
                    GROUP BY username
                    ORDER BY total_cost DESC
                    LIMIT 5
                """
            elif metric == "Environmental Impact":
                query = """
                    SELECT username, SUM(gwp) AS total_gwp
                    FROM chatbot_history
                    GROUP BY username
                    ORDER BY total_gwp DESC
                    LIMIT 5
                """
            elif metric == "Latency":
                query = """
                    SELECT username, AVG(latency) AS avg_latency
                    FROM chatbot_history
                    GROUP BY username
                    ORDER BY avg_latency DESC
                    LIMIT 5
                """
            else:
                raise ValueError("Invalid metric selected.")

            self.cursor.execute(query)
            return self.cursor.fetchall()
        except Exception as e:
            logger.error(f"Error retrieving top users by {metric}: {e}")
            return []

    def get_user_details(self, username: str) -> dict:
        """
        Retrieve basic user information.

        Args:
            username (str): User's username

        Returns:
            dict: User details including username, role, creation date and active status
        """
        try:
            self.cursor.execute(
                """
                SELECT username, role, created_at, is_active
                FROM users
                WHERE username = %s
            """,
                (username,),
            )
            result = self.cursor.fetchone()
            return (
                {
                    "username": result[0],
                    "role": result[1],
                    "created_at": result[2],
                    "is_active": result[3],
                }
                if result
                else {}
            )
        except Exception as e:
            logger.error(f"Erreur récupération utilisateur : {e}")
            return {}

    def get_user_statistics(self, username: str) -> dict:
        """
        Retrieve user usage statistics.

        Args:
            username (str): User's username

        Returns:
            dict: User statistics including chat sessions, queries, costs and performance
        """
        try:
            self.cursor.execute(
                """
                SELECT COUNT(DISTINCT chat_title) AS chat_sessions,
                    COUNT(query) AS total_queries,
                    COALESCE(SUM(query_price), 0) AS money_spent,
                    COALESCE(SUM(gwp), 0) AS environmental_impact,
                    COALESCE(AVG(latency), 0) AS avg_latency
                FROM chatbot_history
                WHERE username = %s
            """,
                (username,),
            )
            result = self.cursor.fetchone()
            return (
                {
                    "Chat Sessions": result[0],
                    "Total Queries": result[1],
                    "Money Spent": result[2],
                    "Environmental Impact": result[3],
                    "Latency": result[4],
                }
                if result
                else {}
            )
        except Exception as e:
            logger.error(f"Erreur récupération statistiques : {e}")
            return {}

    def get_user_feedback(self, username: str) -> list:
        """Récupère l'historique des feedbacks d'un utilisateur."""
        try:
            self.cursor.execute(
                """
                SELECT feedback, comment, timestamp
                FROM chatbot_feedback
                WHERE username = %s
                ORDER BY timestamp DESC
            """,
                (username,),
            )
            return self.cursor.fetchall()
        except Exception as e:
            logger.error(f"Erreur récupération feedbacks : {e}")
            return []

    #################### PAGE ADMIN QUIZZ ############################
    def get_total_quiz_count(self):
        """
        Retrieve user feedback history.

        Args:
            username (str): User's username to fetch feedback for

        Returns:
            list: List of tuples containing (feedback, comment, timestamp)
                Empty list if no feedback or error occurs
        """
        try:
            self.cursor.execute("SELECT COUNT(*) FROM quiz_questions")
            return self.cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du nombre de quiz : {e}")
            return 0

    def get_average_answers_per_quiz(self):
        """Retourne le nombre moyen de réponses par quiz"""
        try:
            self.cursor.execute(
                """
                SELECT AVG(answer_count) 
                FROM (
                    SELECT COUNT(*) as answer_count 
                    FROM quiz_responses 
                    GROUP BY quiz_id
                ) as subquery
            """
            )
            result = self.cursor.fetchone()[0]
            return round(result, 2) if result else 0
        except Exception as e:
            logger.error(
                f"Erreur lors de la récupération du nombre moyen de réponses : {e}"
            )
            return 0

    def get_quiz_success_rate(self):
        """Retourne le taux de réussite moyen (%) des utilisateurs"""
        try:
            self.cursor.execute(
                """
                SELECT 
                    (COUNT(*) FILTER (WHERE is_correct = true) * 100.0) / COUNT(*) 
                FROM quiz_responses
            """
            )
            result = self.cursor.fetchone()[0]
            return round(result, 2) if result else 0
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du taux de réussite : {e}")
            return 0

    def get_top_users_by_success(self, limit=5):
        """Retourne les utilisateurs avec le meilleur taux de réussite"""
        try:
            self.cursor.execute(
                f"""
                SELECT username, 
                       (COUNT(*) FILTER (WHERE is_correct = true) * 100.0) / COUNT(*) AS success_rate
                FROM quiz_responses
                GROUP BY username
                ORDER BY success_rate DESC
                LIMIT {limit}
            """
            )
            return self.cursor.fetchall()  # Liste de tuples (username, success_rate)
        except Exception as e:
            logger.error(
                f"Erreur lors de la récupération des meilleurs utilisateurs : {e}"
            )
            return []

    def get_user_quiz_responses(self, username: str) -> list:
        """
        Retrieve user's quiz response history.

        Args:
            username (str): Username to fetch quiz responses for

        Returns:
            list: List of tuples containing
                (question, user_answer, correct_answer, is_correct, answered_at)
                Empty list if no responses found or error occurs
        """
        try:
            self.cursor.execute(
                """
                SELECT q.question, r.user_answer, q.correct_answer, r.is_correct, r.answered_at
                FROM quiz_responses r
                JOIN quiz_questions q ON r.quiz_id = q.quiz_id
                WHERE r.username = %s
                ORDER BY r.answered_at DESC
            """,
                (username,),
            )
            return self.cursor.fetchall()
        except Exception as e:
            logger.error(f"Erreur récupération réponses quiz : {e}")
            return []


db = SQLDatabase(db_name="poc_rag")
