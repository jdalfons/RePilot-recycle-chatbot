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

from tools.llm_metrics import get_energy_usage, get_price_query

@functools.lru_cache(maxsize=100)  # 🔹 Stocke jusqu'à 100 réponses générées
def cached_process_quiz_question(username: str, question: str, correct_answer: str):
    """
    Vérifie si la question a déjà été traitée récemment.
    Si oui, récupère la réponse stockée sans refaire appel à l'API.
    
    Args:
        question (str): La question du quiz.
        correct_answer (str): La réponse correcte.
    
    Returns:
        dict: Contenant la question reformulée, la réponse correcte et deux fausses réponses.
    """
    return SQLDatabase.process_quiz_question(username,question, correct_answer)

@st.cache_resource
def get_db_connection():
    """ Établit une connexion persistante avec PostgreSQL. """
    return psycopg2.connect(
        dbname=os.getenv('POSTGRES_DBNAME', 'llm'),
        user=os.getenv('POSTGRES_USER', 'llm'),
        password=os.getenv('POSTGRES_PASSWORD', 'llm'),
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=int(os.getenv('POSTGRES_PORT', "32003"))
    )
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
        Initialise la connexion à la base de données PostgreSQL.

        Args:
            db_name (str): Nom de la base de données.
        """
        self.con = get_db_connection()
        self.cursor = self.con.cursor()
        self.db_name = db_name
        self.initialize_database()


    def initialize_database(self):
        """Vérifie et initialise les tables si elles n'existent pas."""
        try:
            self.cursor.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='public';"
            )
            existing_tables = {table[0] for table in self.cursor.fetchall()}
            required_tables = {"chatbot_history", "chatbot_feedback", "quiz_questions", "quiz_responses", "users"}

            missing_tables = required_tables - existing_tables
            if missing_tables:
                print(f"⚠️ Création des tables manquantes : {missing_tables}")
                with open('sql/init.sql', 'r') as file:
                    sql_script = file.read()
                    self.cursor.execute(sql_script)
                    self.con.commit()
                    print("✅ Base de données initialisée avec succès !")
        except Exception as e:
            print(f"❌ Erreur lors de l'initialisation de la base de données : {e}")
            self.con.rollback()

    def add_query(self, query_id: str, query: str, answer: str, embedding_model: str, 
                  generative_model: str, context: str, safe: bool, latency: float, 
                  completion_tokens: int, prompt_tokens: int, query_price: float, 
                  energy_usage: float, gwp: float, username: str = "user"):
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
            self.cursor.execute(insert_query, (
                query_id, query, answer, embedding_model, generative_model, context,
                safe, latency, completion_tokens, prompt_tokens, query_price,
                energy_usage, gwp, username
            ))
            self.con.commit()
            print("✅ Query ajoutée avec succès.")
        except Exception as e:
            print(f"❌ Erreur lors de l'ajout de la requête : {e}")
            self.con.rollback()


    def log_llm_call(self, username: str, query: str, response: str, generative_model: str, 
                  energy_usage: float, gwp: float,
                 completion_tokens: int, prompt_tokens: int, query_price: float, 
                 execution_time_ms: float):
        """
        Enregistre les métriques d'un appel au LLM dans la base de données.

        Args:
            username (str): Nom de l'utilisateur.
            query (str): Question envoyée au LLM.
            response (str): Réponse générée par le LLM.
            generative_model (str): Modèle utilisé.
            context (str): Contexte de l'appel ('quiz' ou 'chatbot').
            energy_usage (float): Consommation d'énergie.
            gwp (float): Potentiel de réchauffement global.
            completion_tokens (int): Nombre de tokens générés.
            prompt_tokens (int): Nombre de tokens envoyés.
            query_price (float): Coût estimé de l'appel LLM.
            execution_time_ms (float): Temps total d'exécution de la requête.
        """
        insert_query = """
        INSERT INTO llm_logs_quiz (
            username, query, response, generative_model , energy_usage, gwp, 
             completion_tokens, prompt_tokens, query_price, execution_time_ms
        ) VALUES (%s, %s, %s, %s,  %s, %s, %s, %s, %s, %s);
        """
        try:
            self.cursor.execute(insert_query, (
                username, query, response, generative_model, energy_usage, gwp, 
                 completion_tokens, prompt_tokens, query_price, execution_time_ms
            ))
            self.con.commit()
            print(f"✅ Log enregistré pour {username}.")
        except Exception as e:
            print(f"❌ Erreur lors de l'enregistrement du log LLM Quiz")
            self.con.rollback()

    # def save_feedback(self, query_id: str, username: str, feedback: str, comment: str = None):
    #     """Enregistre le feedback de l'utilisateur."""
    #     try:
    #         insert_query = """
    #         INSERT INTO chatbot_feedback (query_id, username, feedback, comment, timestamp)
    #         VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
    #         """
    #         self.cursor.execute(insert_query, (query_id, username, feedback, comment))
    #         self.con.commit()
    #         print(f"✅ Feedback ajouté pour {query_id} : {feedback}")
    #     except Exception as e:
    #         print(f"❌ Erreur lors de l'ajout du feedback : {e}")
    #         self.con.rollback()

    @staticmethod
    def process_quiz_question(username: str,question: str, correct_answer: str):
        """
        Unifie tous les appels au LLM pour :
        - Vérifier la pertinence
        - Reformuler la question en précisant la ville
        - Générer la bonne réponse reformulée
        - Produire deux fausses réponses réalistes
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
            - Une réponse de taille proche de deux fausses réponses.
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
                response_format = {
                "type": "json_object",
            }
            )

            json_response = json.loads(response["choices"][0]["message"]["content"])

            # Vérifier si la question est pertinente
            if json_response["pertinent"] == "NON":
                return None  # Exclure les questions non pertinentes

            # **Calcul du score BERTScore**
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Score entre question originale et reformulée
            _, _, F1_question = score(
                [json_response["question_reformulee"]], [question], lang="fr", device=device
            )

            # Score entre réponse correcte originale et reformulée
            _, _, F1_reponse = score(
                [json_response["reponse_courte"]], [correct_answer], lang="fr", device=device
            )

            # Score entre fausses réponses et bonne réponse
            _, _, F1_fake1 = score(
                [json_response["fausse_reponse_1"]], [json_response["reponse_courte"]], lang="fr", device=device
            )
            _, _, F1_fake2 = score(
                [json_response["fausse_reponse_2"]], [json_response["reponse_courte"]], lang="fr", device=device
            )

            # Vérifier si la reformulation de la question et de la réponse sont de qualité
            seuil_pertinence = 0.6  # Seuil pour accepter une reformulation correcte

            if F1_question.item() < seuil_pertinence or F1_reponse.item() < seuil_pertinence:
                print(f"⚠️ Reformulation jugée trop différente ({F1_question.item():.2f}, {F1_reponse.item():.2f})")
                return None  # Exclure la question si la reformulation est trop différente

            # Vérifier si les fausses réponses sont trop proches de la bonne réponse
            if F1_fake1.item() > 0.8 and F1_fake2.item() > 0.8:
                print(f"⚠️ Mauvaises réponses trop proches ({F1_fake1.item():.2f}, {F1_fake2.item():.2f})")
                return None  # Exclure cette génération
            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            prompt_tokens = int(response["usage"]["prompt_tokens"])
            completion_tokens = int(response["usage"]["completion_tokens"])
            query_price = get_price_query("mistral-large-latest", prompt_tokens, completion_tokens)
            energy_usage, gwp = get_energy_usage(response)            
            
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
            execution_time_ms=latency_ms
            )

            return {
                    "question_reformulee": json_response["question_reformulee"],
                    "correct_answer": json_response["reponse_courte"],
                    "fake_answers": [json_response["fausse_reponse_1"], json_response["fausse_reponse_2"]]
                }

        except Exception as e:
            print(f"❌ Erreur lors du traitement de la question : {e}")
            return None
    
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
            self.cursor.execute("SELECT COUNT(*) FROM users WHERE username = %s;", (username,))
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
                (username, limit + 5),
            )
            questions = self.cursor.fetchall()

            if not questions:
                return [{"message": "⚠️ Aucun historique de questions trouvé pour cet utilisateur."}]

            quiz_data = []
            for query, correct_answer in questions:
                # 🔥 Générer la version courte de la bonne réponse + 2 fausses réponses
                processed_data = cached_process_quiz_question(username, query, correct_answer)
                if processed_data:
                            quiz_data.append({
                                "question": processed_data["question_reformulee"],
                                "correct_answer": processed_data["correct_answer"],
                                "fake_answers": processed_data["fake_answers"]
                            })

            return quiz_data

        except Exception as e:
            self.con.rollback()  # ✅ Rétablir la base en cas d’erreur
            print(f"❌ Erreur lors de la récupération des questions du quiz : {e}")
            return []

db = SQLDatabase(db_name="poc_rag")