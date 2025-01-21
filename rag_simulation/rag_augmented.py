from typing import Any
import uuid

from ecologits import EcoLogits

from dotenv import load_dotenv, find_dotenv
import litellm
import numpy as np
from numpy.typing import NDArray

from database.db_management import db
from rag_simulation.schema import Query
from guardrail.service import guardrail_instance
from rag_simulation.corpus_ingestion import BDDChunks
from rag_simulation.wrapper import track_latency

load_dotenv(find_dotenv())


class AugmentedRAG:
    """A class for performing an advanced RAG process.

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
        Initializes the AugmentedRAG class with the provided parameters.

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
        self.db = db

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

    def _get_price_query(
        self, model: str, input_token: int, output_token: int
    ) -> float:
        """
        Calculates the price for the query based on the model and token usage.

        Args:
            model (str): The model name for which to calculate the cost.
            input_token (int): The number of input tokens used.
            output_token (int): The number of output tokens used.

        Returns:
            float: The total cost for the query based on token usage.
        """
        dict_price = {
            "ministral-8b-latest": {"input": 0.10, "output": 0.10},
            "ministral-3b-latest": {"input": 0.04, "output": 0.04},
            "codestral-latest": {"input": 0.20, "output": 0.60},
            "mistral-large-latest": {"input": 2, "output": 6},
        }
        price = dict_price[model]
        return ((price["input"] / 10**6) * input_token) + (
            (price["output"] / 10**6) * output_token
        )

    def _get_energy_usage(self, response: litellm.ModelResponse):
        """
        Extracts energy usage and global warming potential (GWP) from the response.

        Parameters:
            response (litellm.ModelResponse): The model response containing impact data.

        Returns:
            tuple: A tuple (energy_usage, gwp) if impacts are present, otherwise (None, None).
        """
        if hasattr(response, "impacts"):
            try:
                energy_usage = getattr(
                    response.impacts.energy.value, "min", response.impacts.energy.value
                )
            except AttributeError:
                energy_usage = None

            try:
                gwp = getattr(
                    response.impacts.gwp.value, "min", response.impacts.gwp.value
                )
            except AttributeError:
                gwp = None

            return energy_usage, gwp

        return None, None

    @track_latency
    def _generate(self, prompt_dict: list[dict[str, str]]) -> litellm.ModelResponse:
        """
        Generates a response from the LLM using the provided prompt.

        Args:
            prompt_dict (List[Dict[str, str]]): The prompt messages to send to the model.

        Returns:
            litellm.ModelResponse: The response generated by the model. The response is wrapped with a latency evaluator and the Ecologits wrapper.
        """
        EcoLogits.init(providers="litellm", electricity_mix_zone="FRA")
        return litellm.completion(
            model=f"mistral/{self.llm}",
            messages=prompt_dict,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )  # type: ignore

    def call_model(
        self, query: str, context: list[str], prompt_dict: list[dict[str, str]]
    ) -> Query:
        """
        Calls the LLM with the given prompt and returns the response as a Query object.

        Args:
            query (str): The input query to send to the LLM.
            context (List[str]): A list of strings providing the context for the LLM.
            prompt_dict (List[Dict[str, str]]): A list of dictionaries containing prompt details.

        Returns:
            Query: A Query object containing the response from the LLM and related data.
        """
        # Generate the response from the LLM using the provided prompt
        chat_response: dict[str, Any] = self._generate(prompt_dict=prompt_dict)
        # Extract relevant information from the response
        latency = chat_response["latency_ms"]
        input_tokens = int(chat_response["result"].usage.prompt_tokens)
        output_tokens = int(chat_response["result"].usage.completion_tokens)
        # Extract energy usage and global warming potential (GWP)
        energy_usage, gwp = self._get_energy_usage(response=chat_response["result"])
        # Calculate the dollar cost for the query
        dollar_cost = self._get_price_query(
            model=self.llm, input_token=input_tokens, output_token=output_tokens
        )
        # Analyze safety of the query
        safe = self.analyse_safety(query=query)
        # Create and return a Query object with all the gathered data
        return Query(
            query_id=str(uuid.uuid4()),
            query=query,
            answer=str(chat_response["result"].choices[0].message.content),
            context="\n".join(context),
            safe=safe,
            energy_usage=energy_usage,
            gwp=gwp,
            latency=latency,
            completion_tokens=output_tokens,
            prompt_tokens=input_tokens,
            query_price=dollar_cost,
            embedding_model=self.bdd.embedding_name,
            generative_model=self.llm,
        )

    def analyse_safety(self, query: str) -> bool:
        """
        Analyzes the safety of a query using a guardrail instance.

        Args:
            query (str): The input query to check for safety.

        Returns:
            bool: True if the query is safe, False otherwise.
        """
        return guardrail_instance.analyze_query(query=query)

    def get_response(self, response: Query) -> str:
        """
        Returns the response from the LLM, or a guardrail warning if the query is flagged as unsafe.

        Args:
            response (Query): The Query object containing the response data.

        Returns:
            str: The answer from the LLM or a safety warning message.
        """
        if response.safe:
            return response.answer
        else:
            return "**Guardrail activé**: Vous semblez vouloir détourner mon comportement! Veuillez reformuler. 🛡"

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
        print("Chunks retrieved!")
        chunks_list: list[str] = chunks["documents"][0]
        print("Building prompt...")
        prompt_rag = self.build_prompt(
            context=chunks_list, history=str(history), query=query
        )
        response = self.call_model(
            query=query, context=chunks_list, prompt_dict=prompt_rag
        )
        db.add_query(query=response)
        return self.get_response(response=response)
