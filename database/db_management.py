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
import torch
from bert_score import score
from pymongo import MongoClient
from typing import Dict, Optional
from tools.llm_metrics import get_energy_usage, get_price_query
from pymongo.errors import ConnectionFailure, OperationFailure
from pymongo.collection import Collection
from typing import List, Dict, Any, Optional


@functools.lru_cache(maxsize=20)
def cached_process_quiz_question(
    username: str, question: str, correct_answer: str
) -> Optional[Dict[str, str]]:
    """
    Checks if the question has been processed recently.
    If yes, retrieves the cached response instead of making a new API call.

    Args:
        username (str): The user's identifier.
        question (str): The quiz question.
        correct_answer (str): The correct answer.

    Returns:
        Optional[Dict[str, str]]: A dictionary containing:
            - "question_reformulee" (str): Reformulated question.
            - "correct_answer" (str): Reformulated correct answer.
            - "fake_answers" (List[str]): Two fake answers.
        Returns None if the question is deemed irrelevant.
    """
    return SQLDatabase.process_quiz_question(username, question, correct_answer)


@st.cache_resource(ttl=600)  # Caches the connection for 10 minutes
def get_db_connection() -> Optional[psycopg2.extensions.connection]:
    """
    Establishes a persistent connection to the PostgreSQL database.

    Returns:
        Optional[psycopg2.extensions.connection]: The database connection object if successful, None otherwise.
    """
    try:
        return psycopg2.connect(
            dbname=os.getenv("POSTGRES_DBNAME", "llm"),
            user=os.getenv("POSTGRES_USER", "llm"),
            password=os.getenv("POSTGRES_PASSWORD", "llm"),
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "32003")),
        )
    except psycopg2.Error as e:
        st.error(f"❌ Database connection failed: {e}")
        return None  # Avoid breaking execution


class MongoDB:
    """
    A MongoDB handler class that provides utilities for managing databases and collections.
    Supports inserting, querying, and initializing from JSON files.
    """

    def __init__(
        self,
        db_name: str,
        collection_name: str,
        data_dir: str = "data",
        host: str = "localhost",
        port: int = 27017,
        uri: Optional[str] = None,
    ):
        """
        Initializes the MongoDB client.

        Args:
            db_name (str): Name of the MongoDB database.
            collection_name (str): Name of the MongoDB collection.
            data_dir (str, optional): Path to the directory containing JSON data. Defaults to "data".
            host (str, optional): MongoDB host. Defaults to "localhost".
            port (int, optional): MongoDB port. Defaults to 27017.
            uri (str): MongoDB connection URI
        """

        self.db_name = db_name
        self.collection = None
        self.collection_name = collection_name
        self.data_dir = data_dir
        self.host = host
        self.port = int(port)
        self.uri = uri
        # If collection is empty, initialize it
        if self.host is not None:
            self.client = MongoClient(self.host, self.port)
        elif self.uri is not None:
            self.client = MongoClient(self.uri)
        else:
            raise ValueError("Please provide either a host address or a connection URI")
        self.db = None

    def create_collection(self, db_name: str, collection_name: str) -> Collection:
        """
        Creates a MongoDB collection if it doesn't exist.

        Args:
            db_name (str): Database name.
            collection_name (str): Collection name.

        Returns:
            Collection: A reference to the created or existing collection.
        """
        db = self.client[db_name]
        if collection_name not in db.list_collection_names():
            print(f"✅ Created new collection: {collection_name}")
        return db[collection_name]

    def insert_item(
        self, db_name: str, collection_name: str, item: Dict[str, Any]
    ) -> None:
        """
        Inserts an item into the specified MongoDB collection.

        If an item with the same "id" and "ville" exists, it will be updated.

        Args:
            db_name (str): Database name.
            collection_name (str): Collection name.
            item (Dict[str, Any]): The item to insert.

        Raises:
            ValueError: If mandatory fields are missing from the item.
        """
        if "id" not in item or "ville" not in item:
            raise ValueError("❌ The item must contain both 'id' and 'ville' fields.")

        collection = self.create_collection(db_name, collection_name)

        try:
            collection.update_one(
                {"id": item["id"], "ville": item["ville"]},  # Check if the item exists
                {"$set": item},  # Update or insert the item
                upsert=True,
            )
            print(f"✅ Inserted/Updated item with ID: {item['id']}")

        except OperationFailure as e:
            print(f"❌ Error inserting/updating item in MongoDB: {e}")

    def query_collection(
        self, db_name: str, collection_name: str, query: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Queries the specified collection and returns the results as a list.

        Args:
            db_name (str): Database name.
            collection_name (str): Collection name.
            query (Dict[str, Any]): Query dictionary.

        Returns:
            List[Dict[str, Any]]: List of matching documents.
        """
        collection = self.create_collection(db_name, collection_name)

        try:
            results = list(collection.find(query))
            print(f"🔍 Found {len(results)} matching documents in '{collection_name}'")
            return results

        except OperationFailure as e:
            print(f"❌ Query failed: {e}")
            return []

    def setup_database(self) -> None:
        """
        Populates the collection with data from JSON files in the specified directory.

        Raises:
            ValueError: If mandatory fields are missing in any JSON record.
        """
        if not os.path.exists(self.data_dir):
            print(f"⚠️ Data directory '{self.data_dir}' does not exist.")
            return

        json_files = [f for f in os.listdir(self.data_dir) if f.endswith(".json")]
        if not json_files:
            print("⚠️ No JSON files found for database setup.")
            return

        for file_name in json_files:
            file_path = os.path.join(self.data_dir, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)

                    if not isinstance(data, list):
                        print(f"⚠️ Skipping {file_name}: Expected a list of records.")
                        continue

                    for item in data:
                        mandatory_fields = [
                            "id",
                            "ville",
                            "type_dechet",
                            "produits",
                            "action",
                            "instructions",
                        ]
                        if not all(field in item for field in mandatory_fields):
                            raise ValueError(
                                f"❌ Missing required fields in item: {item}"
                            )

                        self.insert_item(self.db_name, self.collection_name, item)

                print(f"✅ Successfully loaded data from {file_name}")

            except (json.JSONDecodeError, ValueError) as e:
                print(f"❌ Error loading JSON data from {file_name}: {e}")


class SQLDatabase:
    """
    Handles database interactions for chatbot history, quizzes, and LLM logs.
    """

    def __init__(self, db_name: str):
        """
        Initializes the database connection.

        Args:
            db_name (str): The database name.
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
                    print("✅ Database initialized successfully!")
        except Exception as e:
            print(f"❌ Error initializing database: {e}")
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
    ) -> None:
        """
        Adds a chatbot query and its response to the history table.

        Args:
            query_id (str): Unique identifier for the query.
            query (str): User's query.
            answer (str): Chatbot's response.
            embedding_model (str): Model used for embeddings.
            generative_model (str): Model used for generation.
            context (str): Additional context information.
            safe (bool): Indicates whether the response is safe.
            latency (float): Response time in milliseconds.
            completion_tokens (int): Number of output tokens.
            prompt_tokens (int): Number of input tokens.
            query_price (float): Estimated cost of the query.
            energy_usage (float): Estimated energy consumption.
            gwp (float): Global warming potential impact.
            username (str): User who initiated the query.
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
            print(f"❌ Error adding query: {e}")
            self.con.rollback()

    def log_llm_call(
        self,
        username: str,
        query: str,
        response: str,
        generative_model: str,
        energy_usage: float,
        gwp: float,
        completion_tokens: int,
        prompt_tokens: int,
        query_price: float,
        execution_time_ms: float,
    ) -> None:
        """
        Logs LLM call metrics to the database.

        Args:
            username (str): The user who made the request.
            query (str): The original query.
            response (str): The generated response.
            generative_model (str): The LLM used.
            energy_usage (float): Energy consumption.
            gwp (float): Global warming potential.
            completion_tokens (int): Number of generated tokens.
            prompt_tokens (int): Number of input tokens.
            query_price (float): Estimated cost of the LLM call.
            execution_time_ms (float): Total execution time in milliseconds.
        """
        insert_query = """
        INSERT INTO llm_logs_quiz (
            username, query, response, generative_model, energy_usage, gwp, 
            completion_tokens, prompt_tokens, query_price, execution_time_ms
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        # print(
        #     "username",
        #     username,
        #     "query",
        #     query,
        #     "response",
        #     response,
        #     "generative_model",
        #     generative_model,
        #     "energy_usage",
        #     energy_usage,
        #     "gwp",
        #     gwp,
        #     "completion_tokens",
        #     completion_tokens,
        #     "prompt_tokens",
        #     prompt_tokens,
        #     "query_price",
        #     query_price,
        #     "execution_time_ms",
        #     execution_time_ms,
        # )
        try:
            self.cursor.execute(
                insert_query,
                (
                    username,
                    query,
                    response,
                    generative_model,
                    energy_usage,
                    gwp,
                    completion_tokens,
                    prompt_tokens,
                    query_price,
                    execution_time_ms,
                ),
            )
            self.con.commit()
            print(f"✅ LLM log recorded for {username}.")
        except Exception as e:
            print("❌ Error logging LLM Quiz Call")
            self.con.rollback()

    def save_feedback(
        self, query_id: str, username: str, feedback: str, comment: str = None
    ) -> bool:
        """Enregistre le feedback de l'utilisateur."""
        try:
            self.cursor.execute(
                insert_query,
                (
                    username,
                    query,
                    response,
                    generative_model,
                    energy_usage,
                    gwp,
                    completion_tokens,
                    prompt_tokens,
                    query_price,
                    execution_time_ms,
                ),
            )
            self.con.commit()
            print(f"✅ Feedback ajouté pour {query_id} : {feedback}")
            return True
        except Exception as e:
            print("❌ Error logging LLM Quiz Call")
            self.con.rollback()
            return False

    @staticmethod
    def process_quiz_question(
        username: str, question: str, correct_answer: str
    ) -> Optional[Dict[str, Any]]:
        """
        Processes a quiz question by:
        - Verifying its relevance to recycling and waste management.
        - Reformulating the question with a city reference if relevant.
        - Generating a concise and accurate correct answer.
        - Producing two misleading but plausible incorrect answers.
        - Logging the LLM call for tracking purposes.

        Args:
            username (str): The user requesting the quiz.
            question (str): The original quiz question.
            correct_answer (str): The detailed correct answer.

        Returns:
            Optional[Dict[str, Any]]: Processed question data including:
                - "question_reformulee": Reformulated question with city reference.
                - "correct_answer": Shortened, clear correct answer.
                - "fake_answers": Two realistic but incorrect answers.
                Returns None if the question is deemed irrelevant or the generated answers fail validation.
        """
        prompt = f"""
        Tu es un expert en tri et recyclage. Analyse la question et génère les éléments suivants :

        ❓ **Question originale** : "{question}"
        ✅ **Réponse correcte détaillée** : "{correct_answer}"

        📌 **Tâches :**
        1️⃣ Vérifie si la question est pertinente pour le tri et le recyclage.
            - Réponds uniquement par "OUI" ou "NON".
        2️⃣ Si pertinent, reformule la question en précisant la ville mentionnée dans "{correct_answer}".
            - La reformulation doit être une question claire et concise adaptée à un quiz.
            - La question reformulée doit inclure une seule ville et pas plus.
        3️⃣ Reformule une **réponse courte et claire** qui garde le sens exact de la réponse correcte détaillée.
            - Pas plus d'une ou deux phrases.
            - Une réponse de taille similaire aux fausses réponses.
        4️⃣ Fournis **exactement 2 fausses réponses distinctes et réalistes**, mais incorrectes par rapport à la question.

        📌 **Réponds uniquement avec un JSON structuré comme suit :**
        {{
            "pertinent": "OUI" ou "NON",
            "question_reformulee": "Nouvelle question avec ville",
            "reponse_courte": "Réponse correcte reformulée",
            "fausse_reponse_1": "Réponse incorrecte 1",
            "fausse_reponse_2": "Réponse incorrecte 2"
        }}
        """

        start_time = time.time()
        try:
            response = litellm.completion(
                model="mistral/mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=1.0,
                api_key=os.getenv("MISTRAL_API_KEY"),
                response_format={"type": "json_object"},
            )

            json_response = json.loads(response["choices"][0]["message"]["content"])

            if json_response["pertinent"] == "NON":
                return None  # Exclude non-relevant questions

            # **BERTScore Calculation for Validation**
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Score between original and reformulated question
            _, _, F1_question = score(
                [json_response["question_reformulee"]],
                [question],
                lang="fr",
                device=device,
            )

            # Score between original and reformulated correct answer
            _, _, F1_reponse = score(
                [json_response["reponse_courte"]],
                [correct_answer],
                lang="fr",
                device=device,
            )

            # Score between fake answers and correct answer
            _, _, F1_fake1 = score(
                [json_response["fausse_reponse_1"]],
                [json_response["reponse_courte"]],
                lang="fr",
                device=device,
            )
            _, _, F1_fake2 = score(
                [json_response["fausse_reponse_2"]],
                [json_response["reponse_courte"]],
                lang="fr",
                device=device,
            )

            # Filtering based on BERTScore
            relevance_threshold = 0.6  # Threshold for accepting a valid reformulation

            if (
                F1_question.item() < relevance_threshold
                or F1_reponse.item() < relevance_threshold
            ):
                print(
                    f"⚠️ Reformulation rejected (F1 scores: {F1_question.item():.2f}, {F1_reponse.item():.2f})"
                )
                return None  # Exclude question if reformulation is too different

            if F1_fake1.item() > 0.8 and F1_fake2.item() > 0.8:
                print(
                    f"⚠️ Fake answers are too close to the correct one ({F1_fake1.item():.2f}, {F1_fake2.item():.2f})"
                )
                return None  # Exclude this generation

            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            prompt_tokens = int(response["usage"]["prompt_tokens"])
            completion_tokens = int(response["usage"]["completion_tokens"])
            query_price = get_price_query(
                "mistral-large-latest", prompt_tokens, completion_tokens
            )
            energy_usage, gwp = get_energy_usage(response)

            # ✅ Log LLM Call
            db.log_llm_call(
                username=username,
                query=question,
                response=json.dumps(json_response),
                generative_model="mistral-large-latest",
                energy_usage=energy_usage,
                gwp=gwp,
                completion_tokens=completion_tokens,
                prompt_tokens=prompt_tokens,
                query_price=query_price,
                execution_time_ms=latency_ms,
            )

            return {
                "question_reformulee": json_response["question_reformulee"],
                "correct_answer": json_response["reponse_courte"],
                "fake_answers": [
                    json_response["fausse_reponse_1"],
                    json_response["fausse_reponse_2"],
                ],
            }

        except Exception as e:
            print(f"❌ Error processing quiz question: {e}")
            return None

    def get_quiz_questions(self, username: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves a quiz based on user-asked questions and generates incorrect answers using the LLM.

        Args:
            username (str): The user's name.
            limit (int, optional): Number of questions to retrieve. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing:
                - "question": Reformulated question.
                - "correct_answer": Reformulated correct answer.
                - "fake_answers": Two realistic incorrect answers.
        """
        try:
            # ✅ Check if the user exists
            self.cursor.execute(
                "SELECT COUNT(*) FROM users WHERE username = %s;", (username,)
            )
            user_exists = self.cursor.fetchone()[0]

            if user_exists == 0:
                print(f"⚠️ User {username} does not exist in the users table.")
                return [{"message": "⚠️ No account found for this user."}]

            # ✅ Retrieve random questions from history
            self.cursor.execute(
                """
                SELECT query, answer 
                FROM chatbot_history
                WHERE username = %s  
                ORDER BY RANDOM()
                LIMIT %s;
                """,
                (username, limit + 2),  # Fetch more to account for filtering
            )
            questions = self.cursor.fetchall()

            if not questions:
                return [{"message": "⚠️ No question history found for this user."}]

            quiz_data = []
            for query, correct_answer in questions:
                #  Process the quiz question through cached LLM call
                processed_data = cached_process_quiz_question(
                    username, query, correct_answer
                )
                if processed_data:
                    quiz_data.append(
                        {
                            "question": processed_data["question_reformulee"],
                            "correct_answer": processed_data["correct_answer"],
                            "fake_answers": processed_data["fake_answers"],
                        }
                    )

            return quiz_data

        except Exception as e:
            self.con.rollback()  # ✅ Rollback in case of error
            print(f"❌ Error retrieving quiz questions: {e}")
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
