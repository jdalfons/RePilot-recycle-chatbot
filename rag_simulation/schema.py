from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class Query(BaseModel):
    query_id: str = Field(description="Unique ID for the query")
    query: str = Field(..., description="The user query")
    answer: str = Field(..., description="The LLM's answer")
    embedding_model: str = Field(
        ..., description="The embedding model used for the RAG pipeline"
    )
    generative_model: str = Field(
        ..., description="The generative model used for the RAG pipeline"
    )
    context: str = Field(..., description="The retrieved chunks")
    timestamp: datetime = Field(
        default=datetime.now(), description="The timestamp of the query"
    )
    safe: bool = Field(
        default=True,
        description="Whether the query was considered safe by the Guardrail",
    )
    energy_usage: Optional[float] = Field(
        None,
        description="The energy usage estimation to run the inference for the query in kilowatt-hours (kWh)",
    )
    gwp: Optional[float] = Field(
        None,
        description="The global warming potential (GWP) in kilograms of CO2 equivalent (kgCO2eq) estimation to run the inference for the query.",
    )
    query_price: float = Field(
        ..., description="The estimated price ($) or cost of the query answering"
    )
    latency: float = Field(..., description="The latency of the response in seconds")
    completion_tokens: int = Field(
        ..., description="The number of tokens generated in the completion by the model"
    )
    prompt_tokens: int = Field(
        ..., description="The number of tokens in the user's input (prompt)"
    )
