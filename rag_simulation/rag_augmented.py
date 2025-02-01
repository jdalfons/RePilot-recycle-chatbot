import logging
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
        selected_city: str = "Grand Lyon Métropole",
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
            selected_city (str, optional): The selected city for which to retrieve information. Defaults to "Grand Lyon Métropole".
        """
        self.llm = generation_model
        self.bdd = bdd_chunks
        self.top_n = top_n
        self.role_prompt = role_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.db = db
        self.selected_city = selected_city
        self.embedding_name = self.bdd.embedding_name
      
        # Vérification de l'initialisation correcte
        if not self.bdd:
            raise ValueError("❌ L'instance BDDChunks n'est pas initialisée !")
    
    def build_prompt(
        self, context: list[str], history: str, query: str
    ) -> list[dict[str, str]]:
        """
        Builds a prompt string for a conversational agent based on the given context and query.

        Args:
            context (str): The context information, typically extracted from books or other sources.
            history (str): The history of the conversation so far.
            query (str): The user's query or question.

        Returns:
            list[dict[str, str]]: The RAG prompt in the OpenAI format
        """

        # Ajout de contexte pour les villes en fonction de la ville choisi par l'utilisateur
        contexte_couleur_bac_ville = ""
        if self.selected_city == "Grand Lyon Métropole":
            contexte_couleur_bac_ville = """
            L'utilisateur se trouve à Grand Lyon Métropole.
            Bac de couleur Vert pour déchets de type verre
            Bac de couleur Jaune pour déchets de type plastique, papier et carton
            Bac de couleur Gris pour déchets ménagers
            Bac de couleur Marron pour déchets de type alimentaires
            Pour le reste, dîtes a l'utilisateur que c'est non applicable"""

        elif self.selected_city == "Paris":
            contexte_couleur_bac_ville = """
            L'utilisateur se trouve à Paris.
            Trier les papiers, emballages en carton, métal et plastique dans le bac jaune ou Trilib.
            Trier les bouteilles, bocaux et pots en verre dans le bac blanc ou Trilib' ou colonne à verre.
            Trier les déchets alimentaires dans le bac marron.
            Les déchets non triables doivent être jetés dans le bac vert ou gris après vérification (non encombrant, dangereux, médicaments, batteries)."""
        
        context_joined = "\n".join(context)
        system_prompt = self.role_prompt
        history_prompt = f"""
        # Historique de conversation:
        {history}
        """
        context_prompt = f"""
        Tu disposes de la section "Contexte" pour t'aider à répondre aux questions.
        # Contexte: 
        {context_joined + contexte_couleur_bac_ville}
        """
        query_prompt = f"""
        # Question:
        {query}

        # Réponse:
        """
        return [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": history_prompt}, # devrait plutôt être assistant ? (réponses antérieurs du model)
            {"role": "assistant", "content": context_prompt}, # devrait plutôt être assistant ? (maintenir le contexte)
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
            prompt_dict (List[Dict[str, str]]): A list of dictionaries containing prompt details. (role from config.yml and chat history)

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

        query_obj = Query(
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
        embedding_model=self.bdd.embedding_model.__class__.__name__,
        generative_model=self.llm,
    )
        return query_obj
        

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
        Exécute le processus RAG pour générer une réponse.

        Args:
            query (str): Question de l'utilisateur.
            history (dict[str, str]): Historique de conversation.

        Returns:
            str: Réponse générée.
        """
        try:
            results = self.bdd.chroma_db.query(query_texts=[query + f" ville: {self.selected_city}"],
            n_results=self.top_n,) # Mettre n_results limite le rag à 2 résultats pour l'instant)
      
            if not results["documents"]:
                return "❌ Aucun document pertinent trouvé pour répondre à votre question."

            chunks_list = results["documents"][0]
            print("Building prompt...")
            prompt_rag = self.build_prompt(context=chunks_list, history=str(history), query=query)

            query_obj = self.call_model(query=query, context=chunks_list, prompt_dict=prompt_rag)
            
             # ✅ Enregistrement dans la base
            self.db.add_query(
            query_id=query_obj.query_id,
            query=query_obj.query,
            answer=query_obj.answer,
            embedding_model=query_obj.embedding_model,
            generative_model=query_obj.generative_model,
            context=query_obj.context,
            safe=query_obj.safe,
            latency=query_obj.latency,
            completion_tokens=query_obj.completion_tokens,
            prompt_tokens=query_obj.prompt_tokens,
            query_price=query_obj.query_price,
            energy_usage=query_obj.energy_usage,
            gwp=query_obj.gwp,
            )

            return self.get_response(response=query_obj) 

        except Exception as e:
            logging.error(f"❌ Erreur dans AugmentedRAG : {e}")
            return "Une erreur s'est produite lors du traitement de votre requête."

