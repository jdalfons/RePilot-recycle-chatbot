# =======
# config = Config('config.yml')
# config_chatbot = config.get_role_prompt()
# config_pdf = config.get_pdf_path() # Deprecated ?


# load_dotenv(find_dotenv())

# st.set_page_config(
#     page_title="ChatBot",
#     page_icon="🤖",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # On regarde si la ville n'a pas déjà été sélectionné et mise en session
# if "ville_choisi" not in st.session_state:
#     st.session_state["ville_choisi"] = "Paris"

# # Add city selector
# st.session_state["ville_choisi"] = st.radio(
#     "Choisissez votre ville",
#     ["Paris", "Grand Lyon Métropole"],
#     index=0,
#     horizontal=True
# )

# @st.cache_resource  # cache_ressource permet de ne pas avoir à reload la fonction à chaque fois que l'on fait une action sur l'application
# def instantiate_bdd() -> BDDChunks:
#     bdd = BDDChunks(embedding_model="paraphrase-multilingual-MiniLM-L12-v2")
#     bdd()
#     return bdd

# # On clear le cache pour relancer la fonction si la ville a changé
# if 'previous_city' not in st.session_state:
#     st.session_state['previous_city'] = st.session_state["ville_choisi"]

# if st.session_state['previous_city'] != st.session_state["ville_choisi"]:
#     st.cache_resource.clear()
#     st.session_state['previous_city'] = st.session_state["ville_choisi"]


# col1, col2 = st.columns([1, 2])

# with col1:
#     generation_model = st.selectbox(
#         label="Choose your LLM",
#         options=[
#             "ministral-8b-latest",
#             "ministral-3b-latest",
#             "codestral-latest",
#             "mistral-large-latest",
#         ],
#     )

# with col2:
#     role_prompt = st.text_area(
#         label=config_chatbot.get('label'),
#         value=config_chatbot.get('value'),
#     )

# with st.expander("options avancées"):
#     col_max_tokens, col_temperature, _ = st.columns([0.25, 0.25, 0.5])

#     with col_max_tokens:
#         max_tokens = st.select_slider(
#             label="Output max tokens", options=list(range(200, 2000, 50)), value=1000
#         )

#     with col_temperature:
#         range_temperature = [round(x, 2) for x in list(numpy.linspace(0, 1.5, num=51))] 
#         temperature = st.select_slider(label="Temperature", options=range_temperature, value=1.2)

# llm = AugmentedRAG(
#     role_prompt=role_prompt,
#     generation_model=generation_model,
#     bdd_chunks=instantiate_bdd(),
#     top_n=2,
#     max_tokens=max_tokens,
#     temperature=temperature,
#     selected_city=st.session_state["ville_choisi"],
# )

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # On affiche les messages de l'utilisateur et de l'IA entre chaque message
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Si présence d'un input par l'utilisateur,
# if query := st.chat_input(""):
#     # On affiche le message de l'utilisateur
#     with st.chat_message("user"):
#         st.markdown(query)
#     # On ajoute le message de l'utilisateur dans l'historique de la conversation
#     st.session_state.messages.append({"role": "user", "content": query})
#     # On récupère la réponse du chatbot à la question de l'utilisateur
#     response = llm(
#         query=query,
#         history=st.session_state.messages,
#     )
#     # On affiche la réponse du chatbot
#     with st.chat_message("assistant"):
#         st.markdown(response)
#     # On ajoute le message du chatbot dans l'historique de la conversation
#     st.session_state.messages.append({"role": "assistant", "content": response})
#     # On ajoute un bouton pour réinitialiser le chat
# if st.button("Réinitialiser le Chat", type="primary"):
#     st.session_state.messages = []
#     st.rerun()

# >>>>>>> Test
