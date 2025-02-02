"""
Streamlit-based chatbot interface using AugmentedRAG and ChromaDB.
"""

import streamlit as st
import numpy as np
from dotenv import find_dotenv, load_dotenv
from rag_simulation.rag_augmented import AugmentedRAG
from rag_simulation.corpus_ingestion import BDDChunks
from database.db_management import db
from utils import Config


# ✅ Configure Streamlit page settings
st.set_page_config(
    page_title="ChatBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ✅ Load environment variables
load_dotenv(find_dotenv())

# ✅ Load chatbot configuration
CONFIG = Config("config.yml")
CONFIG_CHATBOT = CONFIG.get_role_prompt()


def instantiate_bdd() -> BDDChunks:
    """
    Initializes ChromaDB as a persistent resource to prevent multiple reloads.

    Returns:
        BDDChunks: ChromaDB instance for data retrieval.
    """
    bdd_instance = BDDChunks(embedding_model="paraphrase-multilingual-MiniLM-L12-v2")
    bdd_instance()
    return bdd_instance


# ✅ Manage selected city in session state
if "selected_city" not in st.session_state:
    st.session_state["selected_city"] = "Paris"

st.session_state["selected_city"] = st.radio(
    "Select your city",
    ["Paris", "Grand Lyon Métropole"],
    index=0,
    horizontal=True,
)

# ✅ Clear cache if the city selection changes
if "previous_city" not in st.session_state:
    st.session_state["previous_city"] = st.session_state["selected_city"]

if st.session_state["previous_city"] != st.session_state["selected_city"]:
    st.cache_resource.clear()
    st.session_state["previous_city"] = st.session_state["selected_city"]

# ✅ Select model and advanced parameters
col1, col2 = st.columns([1, 2])

with col1:
    GENERATION_MODEL = st.selectbox(
        label="Choose your LLM model",
        options=[
            "ministral-8b-latest",
            "ministral-3b-latest",
            "codestral-latest",
            "mistral-large-latest",
        ],
    )

with col2:
    ROLE_PROMPT = st.text_area(
        label=CONFIG_CHATBOT.get("label"),
        value=CONFIG_CHATBOT.get("value"),
    )

# ✅ Advanced options
with st.expander("Advanced Options"):
    col_max_tokens, col_temperature, _ = st.columns([0.25, 0.25, 0.5])

    with col_max_tokens:
        MAX_TOKENS = st.select_slider(
            label="Max tokens", options=list(range(200, 2000, 50)), value=1000
        )

    with col_temperature:
        TEMPERATURE = st.select_slider(
            label="Temperature",
            options=[round(x, 2) for x in np.linspace(0, 1.5, num=51)],
            value=1.2,
        )

# ✅ Initialize AugmentedRAG pipeline
LLM = AugmentedRAG(
    role_prompt=ROLE_PROMPT,
    generation_model=GENERATION_MODEL,
    bdd_chunks=instantiate_bdd(),
    top_n=1,
    max_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
    selected_city=st.session_state["selected_city"],
)

# ✅ Manage chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ✅ Process user input and chatbot response
if query := st.chat_input("Ask your question here..."):
    with st.chat_message("user"):
        st.markdown(query)

    st.session_state.messages.append({"role": "user", "content": query})

    # Retrieve chatbot response
    query_obj = LLM(query=query, history=st.session_state.messages)

    response = query_obj if isinstance(query_obj, str) else query_obj.answer

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# ✅ Reset chat session
if st.button("Reset Chat", type="primary"):
    st.session_state.messages = []
    st.rerun()
