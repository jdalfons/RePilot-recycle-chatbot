import json
import re
from typing import List
import nltk
from nltk.tokenize import sent_tokenize
import yaml

nltk.download('punkt')

class JSONProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the JSON processor with configurable chunk parameters.

        Args:
            chunk_size (int): Target size of each chunk in characters
            chunk_overlap (int): Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def read_json(self, file_path: str) -> str:
        """
        Read and extract text from a JSON file.

        Args:
            file_path (str): Path to the JSON file

        Returns:
            str: Extracted text from the JSON
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            text = json.dumps(data)
        return text

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize the extracted text.

        Args:
            text (str): Raw text to clean

        Returns:
            str: Cleaned text
        """
        text = text.lower()
        text = ' '.join(text.split())
        text = re.sub(r'[^a-z0-9\s\.]', '', text)
        text = re.sub(r'\.(?! )', '. ', text)
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
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
        return chunks

    def process_json(self, file_path: str) -> List[str]:
        """
        Process a JSON file and return chunks ready for RAG.

        Args:
            file_path (str): Path to the JSON file

        Returns:
            List[str]: List of processed text chunks
        """
        raw_text = self.read_json(file_path)
        cleaned_text = self.clean_text(raw_text)
        chunks = self.create_chunks(cleaned_text)
        return chunks

class Config:
    def __init__(self, config_path: str):
        """
        Initialize the Config class with the path to the configuration file.

        Args:
            config_path (str): Path to the configuration file
        """
        self.config_path = config_path
        self.config = dict()
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
            
    def get(self, key: str):
        """
        Get a value from the configuration file.

        Args:
            key (str): Key to retrieve from the configuration

        Returns:
            str: Value corresponding to the key
        """
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, None)
            if value is None:
                return None
        return value
    
    def get_config(self):
        """
        Get the entire configuration dictionary.

        Returns:
            dict: Configuration dictionary
        """
        return self.config
            
    def get_role_prompt(self) -> dict:
        """
        Get the role prompt configuration.

        Returns:
            dict: Role prompt configuration with label and value
        """
        role_prompt = self.get('config.role.role_prompt')
        if role_prompt:
            return {
                'label': role_prompt.get('label'),
                'value': role_prompt.get('value')
            }
        return {}
    
    def get_pdf_path(self) -> dict:
        """
        Get the PDF path configuration.

        Returns:
            dict: PDF path configuration with label and value
        """
        pdf_path = self.get('config.pdf.path')
        if pdf_path:
            return {
                'label': pdf_path.get('label'),
                'value': pdf_path.get('value')
            }
        return {}
        