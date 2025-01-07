import PyPDF2
import re
from typing import List, Optional
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the PDF processor with configurable chunk parameters.
        
        Args:
            chunk_size (int): Target size of each chunk in characters
            chunk_overlap (int): Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def read_pdf(self, file_path: str) -> str:
        """
        Read and extract text from a PDF file.
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text from the PDF
        """
        with open(file_path, 'rb') as file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract text from all pages
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text() + ' '
                
        return text

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize the extracted text.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep periods for sentence boundaries
        text = re.sub(r'[^a-z0-9\s\.]', '', text)
        
        # Fix spacing after periods
        text = re.sub(r'\.(?! )', '. ', text)
        
        # Remove multiple periods
        text = re.sub(r'\.{2,}', '.', text)
        
        return text

    def create_chunks(self, text: str) -> List[str]:
        """
        Split the text into chunks of a specified length.
        
        Args:
            text (str): Cleaned text to chunk
            
        Returns:
            List[str]: List of text chunks
        """
        chunk_size = 512
        chunks = []
        
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])
        
        return chunks

    def process_pdf(self, file_path: str) -> List[str]:
        """
        Process a PDF file and return chunks ready for RAG.
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            List[str]: List of processed text chunks
        """
        # Extract text from PDF
        raw_text = self.read_pdf(file_path)
        
        # Clean the text
        cleaned_text = self.clean_text(raw_text)
        
        # Create chunks
        chunks = self.create_chunks(cleaned_text)
        
        return chunks
    
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from typing import List, Dict, Any

class MistralRAG:
    def __init__(self, api_key: str, model_name: str = "mistral-tiny"):
        """
        Initialize the RAG system with Mistral.
        
        Args:
            api_key (str): Mistral API key
            model_name (str): Name of the Mistral model to use
                            Options: mistral-tiny, mistral-small, mistral-medium, mistral-large
        """
        os.environ["MISTRAL_API_KEY"] = api_key
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize Mistral LLM
        self.llm = ChatMistralAI(
            model=model_name,
            temperature=0.7,
            max_tokens=1024
        )
        
        self.pdf_processor = PDFProcessor()
        self.vector_store = None

    def ingest_pdf(self, pdf_path: str) -> None:
        """
        Process a PDF and create a vector store from its contents.
        
        Args:
            pdf_path (str): Path to the PDF file
        """
        # Process the PDF and get chunks
        chunks = self.pdf_processor.process_pdf(pdf_path)
        
        # Create vector store
        self.vector_store = FAISS.from_texts(
            chunks,
            self.embeddings
        )

    def add_pdf(self, pdf_path: str) -> None:
        """
        Add another PDF to the existing vector store.
        
        Args:
            pdf_path (str): Path to the PDF file
        """
        chunks = self.pdf_processor.process_pdf(pdf_path)
        
        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(chunks, self.embeddings)
        else:
            self.vector_store.add_texts(chunks)

    def save_vector_store(self, directory: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            directory (str): Directory to save the vector store
        """
        if self.vector_store is not None:
            self.vector_store.save_local(directory)

    def load_vector_store(self, directory: str) -> None:
        """
        Load a vector store from disk.
        
        Args:
            directory (str): Directory containing the vector store
        """
        self.vector_store = FAISS.load_local(directory, self.embeddings)

    def query(self, 
              question: str, 
              k: int = 4, 
              temperature: float = 0.7,
              max_tokens: int = 1024) -> Dict[str, Any]:
        """
        Query the RAG system using Mistral.
        
        Args:
            question (str): Question to ask
            k (int): Number of relevant chunks to retrieve
            temperature (float): Temperature for response generation
            max_tokens (int): Maximum tokens in response
            
        Returns:
            Dict[str, Any]: Dictionary containing the answer and source chunks
        """
        if self.vector_store is None:
            raise ValueError("No documents have been ingested yet!")

        # Update LLM parameters
        self.llm.temperature = temperature
        self.llm.max_tokens = max_tokens

        # Create proper prompt template
        prompt_template = PromptTemplate(
            template="""Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know. 
            Don't try to make up an answer.
            
            Context: {context}
            
            Question: {question}
            
            Answer: """,
            input_variables=["context", "question"]
        )

        # Create retrieval chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": k}),
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt_template
            }
        )
        
        # Get response
        response = qa_chain.invoke({"query": question})
        
        return {
            "answer": response["result"],
            "source_chunks": [doc.page_content for doc in response["source_documents"]]
        }

    def get_relevant_chunks(self, query: str, k: int = 4) -> List[str]:
        """
        Retrieve the most relevant chunks for a query without generating an answer.
        
        Args:
            query (str): Query to find relevant chunks for
            k (int): Number of chunks to retrieve
            
        Returns:
            List[str]: List of relevant text chunks
        """
        if self.vector_store is None:
            raise ValueError("No documents have been ingested yet!")
            
        docs = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from typing import List, Dict, Any

class MistralRAG:
    def __init__(self, api_key: str, model_name: str = "mistral-tiny"):
        """
        Initialize the RAG system with Mistral.
        
        Args:
            api_key (str): Mistral API key
            model_name (str): Name of the Mistral model to use
                            Options: mistral-tiny, mistral-small, mistral-medium, mistral-large
        """
        os.environ["MISTRAL_API_KEY"] = api_key
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize Mistral LLM
        self.llm = ChatMistralAI(
            model=model_name,
            temperature=0.7,
            max_tokens=1024
        )
        
        self.pdf_processor = PDFProcessor()
        self.vector_store = None

    def ingest_pdf(self, pdf_path: str) -> None:
        """
        Process a PDF and create a vector store from its contents.
        
        Args:
            pdf_path (str): Path to the PDF file
        """
        # Process the PDF and get chunks
        chunks = self.pdf_processor.process_pdf(pdf_path)
        
        # Create vector store
        self.vector_store = FAISS.from_texts(
            chunks,
            self.embeddings
        )

    def add_pdf(self, pdf_path: str) -> None:
        """
        Add another PDF to the existing vector store.
        
        Args:
            pdf_path (str): Path to the PDF file
        """
        chunks = self.pdf_processor.process_pdf(pdf_path)
        
        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(chunks, self.embeddings)
        else:
            self.vector_store.add_texts(chunks)

    def save_vector_store(self, directory: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            directory (str): Directory to save the vector store
        """
        if self.vector_store is not None:
            self.vector_store.save_local(directory)

    def load_vector_store(self, directory: str) -> None:
        """
        Load a vector store from disk.
        
        Args:
            directory (str): Directory containing the vector store
        """
        self.vector_store = FAISS.load_local(directory, self.embeddings)

    def query(self, 
              question: str, 
              k: int = 4, 
              temperature: float = 0.7,
              max_tokens: int = 1024) -> Dict[str, Any]:
        """
        Query the RAG system using Mistral.
        
        Args:
            question (str): Question to ask
            k (int): Number of relevant chunks to retrieve
            temperature (float): Temperature for response generation
            max_tokens (int): Maximum tokens in response
            
        Returns:
            Dict[str, Any]: Dictionary containing the answer and source chunks
        """
        if self.vector_store is None:
            raise ValueError("No documents have been ingested yet!")

        # Update LLM parameters
        self.llm.temperature = temperature
        self.llm.max_tokens = max_tokens

        # Create proper prompt template
        prompt_template = PromptTemplate(
            template="""Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know. 
            Don't try to make up an answer.
            
            Context: {context}
            
            Question: {question}
            
            Answer: """,
            input_variables=["context", "question"]
        )

        # Create retrieval chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": k}),
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt_template
            }
        )
        
        # Get response
        response = qa_chain.invoke({"query": question})
        
        return {
            "answer": response["result"],
            "source_chunks": [doc.page_content for doc in response["source_documents"]]
        }

    def get_relevant_chunks(self, query: str, k: int = 4) -> List[str]:
        """
        Retrieve the most relevant chunks for a query without generating an answer.
        
        Args:
            query (str): Query to find relevant chunks for
            k (int): Number of chunks to retrieve
            
        Returns:
            List[str]: List of relevant text chunks
        """
        if self.vector_store is None:
            raise ValueError("No documents have been ingested yet!")
            
        docs = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]