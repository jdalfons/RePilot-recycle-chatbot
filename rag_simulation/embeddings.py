import os
import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

# Set the cache directory to a writable location
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface_cache"

# Initialize the model once to avoid repeated loading
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def get_embedding(documents: list[str]) -> NDArray[np.float32]:
    """
    Generates embeddings for a list of documents using a pre-trained SentenceTransformer model.

    Args:
        documents (list[str]): A list of strings (documents) for which embeddings are to be generated.

    Returns:
        NDArray: A NumPy array containing the embeddings for each document.
    """
    if isinstance(documents, str):
        documents = [documents]
    return model.encode(documents)
