import os
from pickle import load

import streamlit as st

from rag_simulation.embeddings import get_embedding


class Guardrail:
    """
    A class to handle guardrail analysis based on query embeddings.

    Attributes:
        guardrail (Any): The guardrail model used for predictions.
    """

    def __init__(self):
        """
        Initializes the Guardrail class with a guardrail model instance.
        """
        self.guardrail = guardrail_model

    def analyze_query(self, query: str) -> bool:
        """
        Analyzes the given query to determine if it passes the guardrail check.

        Args:
            query (str): The input query to be analyzed.

        Returns:
            bool: Returns `False` if the query is flagged, `True` otherwise.
        """
        embed_query = get_embedding(documents=[query])
        pred = self.guardrail.predict(embed_query.reshape(1, -1)).item()
        return pred != 1  # Return True if pred is not 1, otherwise False


file_path = "./guardrail/storage/guardrail.pkl"
if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        guardrail_model = load(f)
    # Create an instance of the Guardrail class
    guardrail_instance = Guardrail()

else:
    raise RuntimeError(
        f"Guardrail file not found: {file_path}. Please run the notebook 'notebook_training_gr.ipynb' first."
    )
