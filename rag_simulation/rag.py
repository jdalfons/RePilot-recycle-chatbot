import litellm
from dotenv import load_dotenv, find_dotenv

import numpy as np
from numpy.typing import NDArray

from rag_simulation.corpus_ingestion import BDDChunks

load_dotenv(find_dotenv())


class SimpleRAG:
    """A class for performing a simple RAG process.

    This class utilizes a retrieval process to fetch relevant information from a
    database (or corpus) and then passes it to a generative model for further processing.

    """

    def __init__(
        self,
        generation_model: str,
        role_prompt: str,
        bdd_chunks: BDDChunks,
        max_tokens: int,
        temperature: int,
        top_n: int = 2,
    ) -> None:
        """
        Initializes the SimpleRAG class with the provided parameters.

        Args:
            generation_model (str): The model used for generating responses.
            role_prompt (str): The role of the model as specified by the prompt.
            bdd_chunks (Any): The database or chunks of information used in the retrieval process.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (int): The temperature setting for the generative model.
            top_n (int, optional): The number of top documents to retrieve. Defaults to 2.
        """
        self.llm = generation_model
        self.bdd = bdd_chunks
        self.top_n = top_n
        self.role_prompt = role_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature

    def get_cosim(self, a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
        """
        Calculates the cosine similarity between two vectors.

        Args:
            a (NDArray[np.float32]): The first vector.
            b (NDArray[np.float32]): The second vector.

        Returns:
            float: The cosine similarity between the two vectors.
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_top_similarity(
        self,
        embedding_query: NDArray[np.float32],
        embedding_chunks: NDArray[np.float32],
        corpus: list[str],
    ) -> list[str]:
        """
        Retrieves the top N most similar documents from the corpus based on the query's embedding.

        Args:
            embedding_query (NDArray[np.float32]): The embedding of the query.
            embedding_chunks (NDArray[np.float32]): A NumPy array of embeddings for the documents in the corpus.
            corpus (List[str]): A list of documents (strings) corresponding to the embeddings in `embedding_chunks`.
            top_n (int, optional): The number of top similar documents to retrieve. Defaults to 5.

        Returns:
            List[str]: A list of the most similar documents from the corpus, ordered by similarity to the query.
        """
        cos_dist_list = np.array(
            [
                self.get_cosim(embedding_query, embed_doc)
                for embed_doc in embedding_chunks
            ]
        )
        indices_of_max_values = np.argsort(cos_dist_list)[-self.top_n :][::-1]
        # print(indices_of_max_values)
        return [corpus[i] for i in indices_of_max_values]

    def build_prompt(
        self, context: list[str], history: str, query: str
    ) -> list[dict[str, str]]:
        """
        Builds a prompt string for a conversational agent based on the given context and query.

        Args:
            context (str): The context information, typically extracted from books or other sources.
            query (str): The user's query or question.

        Returns:
            list[dict[str, str]]: The RAG prompt in the OpenAI format
        """
        context_joined = "\n".join(context)
        system_prompt = self.role_prompt
        history_prompt = f"""
        # Historique de conversation:
        {history}
        """
        context_prompt = f"""
        Tu disposes de la section "Contexte" pour t'aider à répondre aux questions.
        # Contexte: 
        {context_joined}
        """
        query_prompt = f"""
        # Question:
        {query}

        # Réponse:
        """
        return [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": history_prompt},
            {"role": "system", "content": context_prompt},
            {"role": "user", "content": query_prompt},
        ]

    def _generate(self, prompt_dict: list[dict[str, str]]) -> litellm.ModelResponse:
        response = litellm.completion(
            model=f"mistral/{self.llm}",
            messages=prompt_dict,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )  # type: ignore

        return response

    def call_model(self, prompt_dict: list[dict[str, str]]) -> str:
        """
        Calls the LLM with the given prompt and returns the response.

        Args:
            prompt_dict (List[Dict[str, str]]): A list of dictionaries where each dictionary represents
                                                a message prompt with a string key and string value.

        Returns:
            str: The response generated by the LLM.
        """

        chat_response: str = self._generate(prompt_dict=prompt_dict)
        return str(chat_response.choices[0].message.content)

    def __call__(self, query: str, history: dict[str, str]) -> str:
        """
        Process a query and return a response based on the provided history and database.

        This method performs the following steps:
        1. Queries the ChromaDB instance to retrieve relevant documents based on the input query.
        2. Constructs a prompt using the retrieved documents, the provided query, and the history.
        3. Sends the prompt to the model for generating a response.

        Args:
            query (str): The user query to be processed.
            history (dict[str, str]): A dictionary containing the conversation history,
                where keys represent user inputs and values represent corresponding responses.

        Returns:
            str: The generated response from the model.
        """
        chunks = self.bdd.chroma_db.query(
            query_texts=[query],
            n_results=self.top_n,
        )
        chunks_list = chunks["documents"][0]
        prompt_rag = self.build_prompt(
            context=chunks_list, history=str(history), query=query
        )
        response = self.call_model(prompt_dict=prompt_rag)
        return response
