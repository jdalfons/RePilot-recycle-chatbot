import timeit
from ecologits import EcoLogits

from guardrail.service import guardrail_instance
from dotenv import load_dotenv, find_dotenv
import litellm
import numpy as np
from numpy.typing import NDArray
from rag_simulation.corpus_ingestion import BDDChunks

load_dotenv(find_dotenv())

# latence de réponse OK dans call_model
# input_tokens OK dans call_model
# output_tokens OK dans call_model
# dollar_cost OK dans call_model
# energy_usage OK dans call_model
# gwp OK dans call_model

class AugmentedRAG:
    """A class for performing a simple AugmentedRAG process.

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
        latency: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        dollar_cost: float = 0.0,
        energy_usage: float = 0.0,
        gwp: float = 0.0,
        result: dict = {},
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
        self.latency = latency
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.dollar_cost = dollar_cost
        self.energy_usage = energy_usage
        self.gwp = gwp
        self.result = result

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
        print(indices_of_max_values)
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

        EcoLogits.init(providers="litellm", electricity_mix_zone="FRA")


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
        # latency, time of the response
        start = timeit.default_timer()
        chat_response: str = self._generate(prompt_dict=prompt_dict)
        end = timeit.default_timer()
        latency = end - start
        print("Latency: ", str(latency))
        self.latency = latency
        # Prix d'un modèle ========================================
        # print(chat_response) # résultat de _generate qui est le résultat de litellm.completion
        # print(chat_response.usage.prompt_tokens) # tokens en input
        self.input_tokens = chat_response.usage.prompt_tokens
        # print(chat_response.usage.completion_tokens) # tokens en output
        self.output_tokens = chat_response.usage.completion_tokens
        self.dollar_cost = self._get_price_query(self.llm, self.input_tokens, self.output_tokens)
        # print(self.llm) # ministral-8b-latest

        # Impact environnemental ========================================
        self._getEcoLogitsImpact(chat_response)
        # print(self.energy_usage)
        # print(self.gwp)
        # return a dict with all response
        self.result["response"] = str(chat_response.choices[0].message.content)
        self.result["latency"] = self.latency
        self.result["dollar_cost"] = self.dollar_cost
        self.result["impact_ecologique"] = self.energy_usage
        self.result["gwp"] = self.gwp
        self.result["input_tokens"] = self.input_tokens
        self.result["output_tokens"] = self.output_tokens

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
        chunks_list: list[str] = chunks["documents"][0]
        prompt_rag = self.build_prompt(
            context=chunks_list, history=str(history), query=query
        )
        response = self.call_model(prompt_dict=prompt_rag)
        return response
    
    def _get_price_query(self, nom_model: str, nb_tokens_input: int, nb_tokens_output: int) -> float:
        """
        Calcule le prix d'une requête en fonction du modèle utilisé et du nombre de tokens en entrée et en sortie.
        """
        dict_prix = { 
            "ministral-8b-latest": {"input": 0.10, "output": 0.10}, 
            "ministral-3b-latest": {"input": 0.04, "output": 0.04}, 
            "codestral-latest": {"input": 0.20, "output": 0.60}, 
            "mistral-large-latest": {"input": 2, "output": 6}, 
        }
        prix = dict_prix[nom_model]["input"] * nb_tokens_input + dict_prix[nom_model]["output"] * nb_tokens_output
        return prix
    

    # Initialize EcoLogits
    def _getEcoLogitsImpact(self, response:litellm.ModelResponse) -> float:
        """
        Get the estimated environmental impacts of the inference.
        """
        # Get the energy usage in kWh
        self.energy_usage = getattr(response.impacts.energy.value, "min", response.impacts.energy.value)
        # Get the global warming potential in kgCO2eq
        self.gwp = getattr(response.impacts.gwp.value, "min", response.impacts.gwp.value)
        return self.energy_usage, self.gwp

    def analyse_safety(self, query: str) -> bool:
        """
        Analyzes the safety of a query using a guardrail instance.

        Args:
            query (str): The input query to check for safety.

        Returns:
            bool: True if the query is safe, False otherwise.
        """
        return guardrail_instance.analyze_query(query=query)




