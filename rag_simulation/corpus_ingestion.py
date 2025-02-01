import os
import re
import logging
import chromadb
import uuid
from tqdm import tqdm
from dotenv import load_dotenv

from utils import JSONProcessor
from database.db_management import MongoDB
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

# Charger les variables d'environnement
load_dotenv()
MONGO_HOST = os.getenv('MONGO_HOST', 'localhost')

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

    def get_documents(self, collection: str = 'dechets', database: str = 'rag') -> tuple[list[str], list[str]]:
        """
        Récupère les documents de MongoDB.

        Args:
            collection (str, optional): Nom de la collection MongoDB. Defaults to 'dechets'.
            database (str, optional): Nom de la base MongoDB. Defaults to 'rag'.

        Returns:
            tuple[list[str], list[str]]: Liste des documents texte et des IDs.
        """
        mongo_db = MongoDB(db_name=database, collection_name=collection, host=MONGO_HOST)
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
            raise RuntimeError("❌ La collection ChromaDB n'est pas initialisée !")

        if not list_chunks:
            logging.error("❌ Aucun chunk de texte à insérer !")
            return

        # Générer un UUID pour chaque document
        chroma_ids = [str(uuid.uuid4()) for _ in ids]

        # Générer explicitement les embeddings avant l'ajout dans ChromaDB
        embeddings = self.embedding_model.encode(list_chunks, show_progress_bar=True)

        # Ajouter les documents en batch
        for i in tqdm(range(0, len(list_chunks), batch_size), desc="Ajout des embeddings dans ChromaDB"):
            batch_chunks = list_chunks[i:i + batch_size]
            batch_ids = chroma_ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]

            logging.info(f"📤 Ajout du batch {i} à {i + batch_size} ({len(batch_chunks)} documents)")

            self.chroma_db.add(
                documents=batch_chunks,
                ids=batch_ids,
                embeddings=batch_embeddings
            )

        logging.info(f"✅ {len(list_chunks)} documents ajoutés à ChromaDB avec succès !")

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
            # Récupérer les documents et les IDs
            corpus, ids = self.get_documents()
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

            # Suppression et recréation de la collection ChromaDB
            self.reset_chroma_collection()

            # Ajout des embeddings
            self.add_embeddings(list_chunks=cleaned_texts, ids=ids)

            logging.info("✅ Pipeline d'ingestion terminé avec succès !")

        except Exception as e:
            logging.error(f"❌ Erreur lors du pipeline d'ingestion : {e}")
            raise
