import logging
import os
from pickle import load

import streamlit as st

from rag_simulation.embeddings import get_embedding, get_embedding_backend

EMBEDDING_BACKEND = get_embedding_backend()


class Guardrail:
    """
    A class to handle guardrail analysis based on query embeddings.

    Attributes:
        guardrail (Any): The guardrail model used for predictions.
    """

    KEYWORD_DENYLIST = {
        "attack",
        "bomb",
        "exploit",
        "fraud",
        "hack",
        "kill",
        "terror",
        "weapon",
    }

    def __init__(self):
        """
        Initializes the Guardrail class with a guardrail model instance.
        """
        self.guardrail = guardrail_model
        self._fallback_logged = False

    def analyze_query(self, query: str) -> bool:
        """
        Analyzes the given query to determine if it passes the guardrail check.

        Args:
            query (str): The input query to be analyzed.

        Returns:
            bool: Returns `False` if the query is flagged, `True` otherwise.
        """
        if self.guardrail is not None and EMBEDDING_BACKEND.uses_sentence_transformer():
            embed_query = get_embedding(documents=[query])
            pred = self.guardrail.predict(embed_query.reshape(1, -1)).item()
            return pred != 1  # Return True if pred is not 1, otherwise False

        if not self._fallback_logged:
            logging.info(
                "Guardrail model running in keyword fallback mode because the sentence-transformer embeddings are unavailable."
            )
            self._fallback_logged = True
        return self._keyword_based_check(query)

    def _keyword_based_check(self, query: str) -> bool:
        """Simple keyword-based guardrail used when embeddings are unavailable."""

        lowered_query = query.lower()
        for keyword in self.KEYWORD_DENYLIST:
            if keyword in lowered_query:
                logging.warning(
                    "Guardrail fallback blocked query '%s' due to keyword '%s'",
                    query,
                    keyword,
                )
                return False
        return True


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
