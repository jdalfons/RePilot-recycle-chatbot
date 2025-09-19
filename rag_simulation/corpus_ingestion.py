import os
import re
import logging
import chromadb
import uuid
from typing import Optional
from tqdm import tqdm
from dotenv import load_dotenv

from utils import JSONProcessor
from database.db_management import MongoDB
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
# Charger les variables d'environnement
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", None)
MONGO_HOST = os.getenv("MONGO_HOST", "localhost") or "localhost"
try:
    MONGO_PORT = int(os.getenv("MONGO_PORT", "27017"))
except ValueError:
    logging.warning(
        "⚠️ Invalid MONGO_PORT provided. Falling back to default 27017."
    )
    MONGO_PORT = 27017

class BDDChunks:
    """
    Classe pour gérer l'ingestion de données depuis MongoDB vers ChromaDB avec génération d'embeddings.
    """

    def __init__(self, embedding_model: str):
        """
        Initialise l'instance BDDChunks.

        Args:
            embedding_model (str): Modèle utilisé pour générer les embeddings.
        """
        self.client = chromadb.PersistentClient(
            path="./chromadb",
            settings=Settings(anonymized_telemetry=False)
        )
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_name = embedding_model
        self.chroma_db = None  # La collection sera créée dynamiquement
        self.collection_name = "dechets_collection"
        self.path = "dechets"
        self.embeddings = SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

    def reset_chroma_collection(self) -> None:
        """
        Supprime l'ancienne collection pour éviter les doublons et la recrée proprement.
        """
        try:
            self.client.delete_collection(self.collection_name)
            logging.info(f"🗑️ Collection ChromaDB '{self.collection_name}' supprimée.")
        except Exception:
            logging.warning(f"⚠️ Impossible de supprimer la collection '{self.collection_name}', peut-être inexistante.")

        # Création de la nouvelle collection
        self.chroma_db = self.client.get_or_create_collection(name=self.collection_name)
        logging.info(f"✅ Nouvelle collection ChromaDB créée : {self.collection_name}")

    def get_documents(
        self,
        host: Optional[str],
        port: int,
        uri: Optional[str],
        collection: str = 'dechets',
        database: str = 'rag'
    ) -> tuple[list[str], list[str]]:
        """
        Récupère les documents de MongoDB.

        Args:
            collection (str, optional): Nom de la collection MongoDB. Defaults to 'dechets'.
            database (str, optional): Nom de la base MongoDB. Defaults to 'rag'.

        Returns:
            tuple[list[str], list[str]]: Liste des documents texte et des IDs.
        """
        mongo_db = MongoDB(
            db_name=database,
            collection_name=collection,
            host=host,
            port=port,
            uri=uri,
        )
        all_documents = mongo_db.query_collection(db_name=database, collection_name=collection, query={})

        if not all_documents:
            logging.warning("⚠️ Aucun document trouvé dans MongoDB ! Vérifiez la base de données.")

        ids = [str(doc['_id']) for doc in all_documents]
        documents = [
            " ".join([f"{k}: {v}" for k, v in doc.items() if k not in ['_id', 'id']])
            for doc in all_documents
        ]

        return documents, ids

    def add_embeddings(self, list_chunks: list[str], ids: list[str], batch_size: int = 100) -> None:
        """
        Ajoute des embeddings et métadonnées dans ChromaDB.

        Args:
            list_chunks (list[str]): Liste des textes à indexer.
            ids (list[str]): Liste des IDs MongoDB correspondant aux documents.
            batch_size (int): Nombre de documents traités par lot.
        """
        if self.chroma_db is None:
          
            raise RuntimeError("Must instantiate a ChromaDB collection first!")
        # Divide ids and chunks into lists of max 160
        divided_chunks = [list_chunks[i:i + 160] for i in range(0, len(list_chunks), 160)]
        divided_ids = [ids[i:i + 160] for i in range(0, len(ids), 160)]
        for i in tqdm(range(len(divided_chunks))): 
            self.chroma_db.add(documents=divided_chunks[i], ids=divided_ids[i])

    def _create_collection(self, path: str) -> None:
            """
            Create a new ChromaDB collection for storing embeddings.

            Args:
                path (str): The name of the collection to create in ChromaDB.
            """
            # Tester qu'en changeant de path, on accède pas au reste
            file_name = "a" + os.path.basename(path)[0:50].strip() + "a"
            file_name = re.sub(r"\s+", "-", file_name)
            # Expected collection name that (1) contains 3-63 characters, (2) starts and ends with an alphanumeric character, (3) otherwise contains only alphanumeric characters, underscores or hyphens (-), (4) contains no two consecutive periods (..)
            self.chroma_db = self.client.get_or_create_collection(name=file_name, embedding_function=self.embeddings, metadata={"hnsw:space": "cosine"})  # type: ignore
        
    def __call__(self) -> None:
        """
        Exécute tout le pipeline d'ingestion :
        1. Récupération des documents depuis MongoDB
        2. Nettoyage des textes
        3. Suppression et recréation d'une collection dans ChromaDB
        4. Ajout des embeddings
        """
        logging.info("🚀 Début du pipeline d'ingestion...")

        try:
            host = MONGO_HOST if MONGO_HOST else None
            uri = MONGO_URI if MONGO_URI else None
            port = MONGO_PORT
            # Récupérer les documents et les IDs
            corpus, ids = self.get_documents(host=host, port=port, uri=uri)
            logging.info(f"📂 {len(corpus)} documents récupérés depuis MongoDB")

            # Vérifications pour éviter les erreurs
            if not corpus:
                raise ValueError("❌ Aucun document à encoder ! Vérifiez MongoDB.")
            if not isinstance(corpus[0], str):
                raise TypeError(f"❌ Le format du document est incorrect : {type(corpus[0])}")

            # Nettoyage des textes
            json_processor = JSONProcessor()
            cleaned_texts = [json_processor.clean_text(doc) for doc in corpus]
            logging.info(f"📝 {len(cleaned_texts)} documents nettoyés.")
            self.chunks = cleaned_texts
            self.ids = ids  
            self._create_collection(path=self.path)
            # Ajout des embeddings
            self.add_embeddings(list_chunks=cleaned_texts, ids=ids)

            logging.info("✅ Pipeline d'ingestion terminé avec succès !")

        except Exception as e:
            logging.error(f"❌ Erreur lors du pipeline d'ingestion : {e}")
            raise

