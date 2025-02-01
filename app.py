import streamlit as st
# Configuration de la page Streamlit
st.set_page_config(
    page_title="ChatBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)
from dotenv import find_dotenv, load_dotenv
import numpy as np


# Importation des modules du chatbot
from rag_simulation.rag_augmented import AugmentedRAG
from rag_simulation.corpus_ingestion import BDDChunks
from database.db_management import db
from utils import Config


# Charger la configuration
config = Config('config.yml')
config_chatbot = config.get_role_prompt()
# Charger les variables d’environnement
load_dotenv(find_dotenv())


# On regarde si la ville n'a pas déjà été sélectionné et mise en session
if "ville_choisi" not in st.session_state:
    st.session_state["ville_choisi"] = "Paris"

# Add city selector
st.session_state["ville_choisi"] = st.radio(
    "Choisissez votre ville",
    ["Paris", "Grand Lyon Métropole"],
    index=0,
    horizontal=True
)

@st.cache_resource
def instantiate_bdd() -> BDDChunks:
    """Initialise la base ChromaDB et évite de la recharger plusieurs fois."""
    bdd = BDDChunks(embedding_model="paraphrase-multilingual-MiniLM-L12-v2")
    bdd()
    return bdd


# On clear le cache pour relancer la fonction si la ville a changé
if 'previous_city' not in st.session_state:
    st.session_state['previous_city'] = st.session_state["ville_choisi"]

if st.session_state['previous_city'] != st.session_state["ville_choisi"]:
    st.cache_resource.clear()
    st.session_state['previous_city'] = st.session_state["ville_choisi"]



col1, col2 = st.columns([1, 2])
with col1:
    generation_model = st.selectbox(
        label="Choisissez votre modèle LLM",
        options=[
            "ministral-8b-latest",
            "ministral-3b-latest",
            "codestral-latest",
            "mistral-large-latest",
        ],
    )

with col2:
    role_prompt = st.text_area(
        label=config_chatbot.get('label'),
        value=config_chatbot.get('value'),
    )

# Options avancées
with st.expander("Options avancées"):
    col_max_tokens, col_temperature, _ = st.columns([0.25, 0.25, 0.5])

    with col_max_tokens:
        max_tokens = st.select_slider(
            label="Nombre max de tokens", options=list(range(200, 2000, 50)), value=1000
        )

    with col_temperature:
        range_temperature = [round(x, 2) for x in np.linspace(0, 1.5, num=51)]
        temperature = st.select_slider(label="Température", options=range_temperature, value=1.2)

# Initialisation du pipeline AugmentedRAG
llm = AugmentedRAG(
    role_prompt=role_prompt,
    generation_model=generation_model,
    bdd_chunks=instantiate_bdd(),
    top_n=1,
    max_tokens=max_tokens,
    temperature=temperature,
    selected_city=st.session_state["ville_choisi"],
)

# Gestion de l'historique des messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher l’historique des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialisation de `response` et `query_id` pour éviter les erreurs
response = None
query_id = None

# ✅ Gestion des feedbacks avec `st.session_state`
if "feedback_data" not in st.session_state:
    st.session_state.feedback_data = {}

# Si l'utilisateur pose une question
if query := st.chat_input("Posez votre question ici..."):
    with st.chat_message("user"):
        st.markdown(query)

    st.session_state.messages.append({"role": "user", "content": query})

    # Récupération de la réponse du chatbot
    query_obj = llm(
        query=query,
        history=st.session_state.messages,
    )

    if isinstance(query_obj, str):
        response = query_obj
    else:
        response = query_obj.answer
        query_id = query_obj.query_id  # ✅ Assurer que query_id est bien récupéré
        st.session_state.feedback_data[query_id] = {"feedback": None, "comment": ""}  # Stockage temporaire

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# ✅ Gestion du feedback utilisateur AVEC bouton de validation
if query_id:
    with st.expander("📢 Donnez votre avis sur cette réponse :"):
        selected_feedback = st.radio(
            "Cette réponse était-elle utile ?",
            options=["Non spécifié", "Utile", "Inutile"],
            horizontal=True,
            key=f"feedback_radio_{query_id}",
            index=0 if st.session_state.feedback_data[query_id]["feedback"] is None else 
            ["Non spécifié", "Utile", "Inutile"].index(st.session_state.feedback_data[query_id]["feedback"]),
        )

        comment = st.text_area(
            "Ajoutez un commentaire sur la réponse (facultatif) :",
            key=f"comment_{query_id}",
            value=st.session_state.feedback_data[query_id]["comment"],
        )

        # 🔹 Stocker la sélection temporairement dans `st.session_state`
        st.session_state.feedback_data[query_id]["feedback"] = selected_feedback
        st.session_state.feedback_data[query_id]["comment"] = comment

        # ✅ Le feedback n'est enregistré QUE lorsque l'utilisateur clique sur "Valider mon feedback"
        if st.button("Valider mon feedback", key=f"validate_feedback_{query_id}"):
            if selected_feedback in ["Utile", "Inutile"]:
                success = db.save_feedback(
                    query_id=query_id,
                    username="user",
                    feedback=selected_feedback,
                    comment=comment
                )
                if success:
                    st.success(f"✅ Merci pour votre retour : {selected_feedback}")
                else:
                    st.error("❌ Une erreur est survenue lors de l'enregistrement du feedback.")
            else:
                st.warning("⚠️ Merci de sélectionner 'Utile' ou 'Inutile' avant de valider.")

# Bouton pour réinitialiser le chat
if st.button("Réinitialiser le Chat", type="primary"):
    st.session_state.messages = []
    st.session_state.feedback_data = {}  # 🔹 Réinitialiser les feedbacks aussi
    st.rerun()
