import json
import time
import litellm
import psycopg2
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
def cached_process_quiz_question(username: str, question: str, correct_answer: str) -> Optional[Dict[str, str]]:
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
    ):
        """
        Initializes the MongoDB client.

        Args:
            db_name (str): Name of the MongoDB database.
            collection_name (str): Name of the MongoDB collection.
            data_dir (str, optional): Path to the directory containing JSON data. Defaults to "data".
            host (str, optional): MongoDB host. Defaults to "localhost".
            port (int, optional): MongoDB port. Defaults to 27017.
        """
        try:
            self.client = MongoClient(host, port)
            self.db_name = db_name
            self.collection_name = collection_name
            self.data_dir = data_dir
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]

            # If collection is empty, initialize it
            if self.collection.estimated_document_count() == 0:
                print(f"⚠️ Collection '{collection_name}' is empty. Initializing with data...")
                self.setup_database()
            else:
                print(f"✅ Connected to MongoDB: {db_name}.{collection_name}")

        except ConnectionFailure as e:
            print(f"❌ MongoDB connection error: {e}")

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

    def insert_item(self, db_name: str, collection_name: str, item: Dict[str, Any]) -> None:
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

    def query_collection(self, db_name: str, collection_name: str, query: Dict[str, Any]) -> List[Dict[str, Any]]:
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
                        mandatory_fields = ["id", "ville", "type_dechet", "produits", "action", "instructions"]
                        if not all(field in item for field in mandatory_fields):
                            raise ValueError(f"❌ Missing required fields in item: {item}")

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

    def initialize_database(self) -> None:
        """
        Ensures required tables exist in the database.
        """
        try:
            self.cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
            existing_tables = {table[0] for table in self.cursor.fetchall()}
            required_tables = {"chatbot_history", "chatbot_feedback", "quiz_questions", "quiz_responses", "users"}

            missing_tables = required_tables - existing_tables
            if missing_tables:
                print(f"⚠️ Creating missing tables: {missing_tables}")
                with open("sql/init.sql", "r", encoding="utf-8") as file:
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
            print("✅ Query successfully added.")
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


    @staticmethod
    def process_quiz_question(username: str, question: str, correct_answer: str) -> Optional[Dict[str, Any]]:
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
                [json_response["question_reformulee"]], [question], lang="fr", device=device
            )

            # Score between original and reformulated correct answer
            _, _, F1_reponse = score(
                [json_response["reponse_courte"]], [correct_answer], lang="fr", device=device
            )

            # Score between fake answers and correct answer
            _, _, F1_fake1 = score(
                [json_response["fausse_reponse_1"]], [json_response["reponse_courte"]], lang="fr", device=device
            )
            _, _, F1_fake2 = score(
                [json_response["fausse_reponse_2"]], [json_response["reponse_courte"]], lang="fr", device=device
            )

            # Filtering based on BERTScore
            relevance_threshold = 0.6  # Threshold for accepting a valid reformulation

            if F1_question.item() < relevance_threshold or F1_reponse.item() < relevance_threshold:
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
            query_price = get_price_query("mistral-large-latest", prompt_tokens, completion_tokens)
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
                "fake_answers": [json_response["fausse_reponse_1"], json_response["fausse_reponse_2"]],
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
            self.cursor.execute("SELECT COUNT(*) FROM users WHERE username = %s;", (username,))
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
                processed_data = cached_process_quiz_question(username, query, correct_answer)
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

db = SQLDatabase(db_name="poc_rag") 