import os
import re
import fitz
import chromadb
import tiktoken
from tqdm import tqdm

from utils import JSONProcessor 
from database.db_management import MongoDB

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb.config import Settings

enc = tiktoken.get_encoding("o200k_base")

from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

MONGO_HOST = os.getenv('MONGO_HOST')
POSTGRES_URI = os.getenv('POSTGRES_URI')


class BDDChunks:
    """
    A class to handle operations related to chunking text data, embedding, and storing in a ChromaDB instance.

    This class provides methods to:
    - Read text from PDF files.
    - Split the text into smaller chunks for processing.
    - Create a ChromaDB collection with embeddings for the chunks.
    - Add these chunks and their embeddings to the ChromaDB collection.
    """

    def __init__(self, embedding_model: str):
        """
        Initialize a BDDChunks instance.

        Args:
            embedding_model (str): The name of the embedding model to use for generating embeddings.
            path (str): The file path to the PDF or dataset to process.
        """
        self.path = "dechets"
        self.chunks: list[str] | None = None
        self.client = chromadb.PersistentClient(
            path="./chromadb",
            settings=Settings(anonymized_telemetry=False)
        )
        self.embedding_name = embedding_model
        self.embeddings = SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self.chroma_db = None

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

    def read_pdf(self, file_path: str) -> str:
        """
        Reads the content of a PDF file, excluding the specified number of pages from the start and end.

        Args:
            file_path (str): The path to the PDF file.

        Returns:
            str: The extracted text from the specified pages of the PDF.
        """
        doc = fitz.open(file_path)
        text = str()
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()  # type: ignore
        return text  # type: ignore

    def split_text_into_chunks(self, corpus: str, chunk_size: int = 500) -> list[str]:
        """
        Splits a given text corpus into chunks of a specified size.

        Args:
            corpus (str): The input text corpus to be split into chunks.
            chunk_size (int, optional): The size of each chunk. Defaults to 500.

        Returns:
            list[str]: A list of text chunks.
        """
        tokenized_corpus = enc.encode(corpus)
        chunks = [
            "".join(enc.decode(tokenized_corpus[i : i + chunk_size]))
            for i in tqdm(range(0, len(tokenized_corpus), chunk_size))
        ]

        return chunks

    def add_embeddings(self, list_chunks: list[str], ids:list[str] , batch_size: int = 100) -> None:
        """
        Add embeddings for text chunks to the ChromaDB collection.

        Args:
            list_chunks (list[str]): A list of text chunks to embed and add to the collection.
            batch_size (int, optional): The batch size for adding documents to the collection. Defaults to 100.

        Note:
            ChromaDB supports a maximum of 166 documents per batch.
        """
        if self.chroma_db is None:
            raise RuntimeError("Must instantiate a ChromaDB collection first!")
        if len(list_chunks) < batch_size:
            batch_size_for_chromadb = len(list_chunks)
        else:
            batch_size_for_chromadb = batch_size
        
        
        # Divide ids and chunks into lists of max 160
        divided_chunks = [list_chunks[i:i + 160] for i in range(0, len(list_chunks), 160)]
        divided_ids = [ids[i:i + 160] for i in range(0, len(ids), 160)]
        for i in tqdm(range(len(divided_chunks))): 
            self.chroma_db.add(documents=divided_chunks[i], ids=divided_ids[i])


    def get_documents(self, host: str, collection: str='dechets', database: str='rag') -> tuple[list[str], list[str]]:
        
        mongoDb = MongoDB(
            db_name=database,
            collection_name=collection, 
            host=host
            )
        
        

        all_documents_list = mongoDb.query_collection(database, collection, dict()) # version originale, dict pour tout prendre, est-ce que l'output est le même ?
        ids = [str(doc['_id']) for doc in all_documents_list]
        documents = [" ".join([f"{k}: {v}" for k, v in doc.items() if k != '_id' and k != 'id']) for doc in all_documents_list]
        
        return documents, ids
        
        
    def __call__(self) -> None:
        """
        Execute the entire process of reading, chunking, creating a collection, and adding embeddings.

        This method:
        1. Reads the text from the specified PDF file.
        2. Splits the text into chunks.
        3. Creates a ChromaDB collection for storing the embeddings.
        4. Adds the text chunks and their embeddings to the ChromaDB collection.
        """
        
        host = MONGO_HOST if MONGO_HOST != None else 'localhost'

        corpus, ids = self.get_documents(host=host)
        json_processor = JSONProcessor()
        chunks = [json_processor.clean_text(doc) for doc in corpus]
        self.chunks = chunks
        self.ids = ids  
        self._create_collection(path=self.path)
        self.add_embeddings(list_chunks=chunks, ids=ids)
