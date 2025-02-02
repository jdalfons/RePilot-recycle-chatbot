import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from database.db_management import SQLDatabase
from datetime import datetime
from rag_simulation.rag_augmented import AugmentedRAG
from rag_simulation.corpus_ingestion import BDDChunks
from plots.plots import get_line_plot_user


from dotenv import find_dotenv, load_dotenv
from database.db_management import db
from utils import Config




class UserDashboard:
    def __init__(self):
        self.db = SQLDatabase(db_name="poc_rag")
        self.llm = self._init_cached_llm()
        self.init_session_state()

    def init_session_state(self):
        """Initialize session state variables"""
        defaults = {
            "page": "chats",
            "current_chat": f"Chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "chat_history": [],
            "quiz_state": {"current_question": 0, "score": 0},
            "username": st.session_state.get("username", "Anonymous")
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    @staticmethod
    @st.cache_resource
    def _init_cached_llm():
        """Initialize LLM with caching"""
        bdd = BDDChunks(embedding_model="paraphrase-multilingual-MiniLM-L12-v2")
        bdd()
        return AugmentedRAG(
            role_prompt="Assistant RePilot",
            generation_model="ministral-8b-latest",
            bdd_chunks=bdd,
            top_n=2,
            max_tokens=1000,
            temperature=1.2
        )

    def show_quiz(self):
        """Display quiz interface"""
        st.title("📝 Quiz RePilot")
        
        quiz_questions = [
            {
                "question": "Que peut-on recycler dans le bac jaune?",
                "options": ["Verre", "Plastique", "Déchets alimentaires", "Piles"],
                "correct": 1,
                "explanation": "Le plastique va dans le bac jaune pour être recyclé."
            },
            {
                "question": "Quelle est la durée de vie d'une bouteille plastique?",
                "options": ["100 ans", "450 ans", "1000 ans", "50 ans"],
                "correct": 1,
                "explanation": "Une bouteille plastique met environ 450 ans à se dégrader."
            }
        ]

        current_q = st.session_state.quiz_state["current_question"]
        
        if current_q < len(quiz_questions):
            self._display_question(quiz_questions[current_q], current_q)
        else:
            self._show_quiz_results(len(quiz_questions))

    def _display_question(self, question, current_q):
        """Display a single quiz question"""
        st.subheader(f"Question {current_q + 1}")
        st.write(question["question"])
        
        answer = st.radio("Choisissez votre réponse:", 
                         question["options"], 
                         key=f"quiz_{current_q}")
        
        if st.button("Valider"):
            if question["options"].index(answer) == question["correct"]:
                st.success("Correct! " + question["explanation"])
                st.session_state.quiz_state["score"] += 1
            else:
                st.error("Incorrect. " + question["explanation"])
            st.session_state.quiz_state["current_question"] += 1
            st.rerun()

    def _show_quiz_results(self, total_questions):
        """Display quiz results"""
        st.success(f"Quiz terminé! Score: {st.session_state.quiz_state['score']}/{total_questions}")
        if st.button("Recommencer"):
            st.session_state.quiz_state = {"current_question": 0, "score": 0}
            st.rerun()
    
    @staticmethod
    @st.cache_resource
    def instantiate_bdd() -> BDDChunks:
        """Initialise la base ChromaDB et évite de la recharger plusieurs fois."""
        bdd = BDDChunks(embedding_model="paraphrase-multilingual-MiniLM-L12-v2")
        bdd()
        return bdd




    def  show_chats(self ):
        # st.title("💬 Mes conversations")
        # Charger la configuration

        # @st.cache_resource
        # def instantiate_bdd() -> BDDChunks:
        #     """Initialise la base ChromaDB et évite de la recharger plusieurs fois."""
        #     bdd = BDDChunks(embedding_model="paraphrase-multilingual-MiniLM-L12-v2")
        #     bdd()
        #     return bdd
        
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

        # @st.cache_resource
        # def instantiate_bdd() -> BDDChunks:
        #     """Initialise la base ChromaDB et évite de la recharger plusieurs fois."""
        #     bdd = BDDChunks(embedding_model="paraphrase-multilingual-MiniLM-L12-v2")
        #     bdd()
        #     return bdd


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
            bdd_chunks=self.instantiate_bdd(),
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

                # # ✅ Le feedback n'est enregistré QUE lorsque l'utilisateur clique sur "Valider mon feedback"
                # if st.button("Valider mon feedback", key=f"validate_feedback_{query_id}"):
                #     if selected_feedback in ["Utile", "Inutile"]:
                #         success = db.save_feedback(
                #             query_id=query_id,
                #             username="user",
                #             feedback=selected_feedback,
                #             comment=comment
                #         )
                #         if success:
                #             st.success(f"✅ Merci pour votre retour : {selected_feedback}")
                #         else:
                #             st.error("❌ Une erreur est survenue lors de l'enregistrement du feedback.")
                #     else:
                #         st.warning("⚠️ Merci de sélectionner 'Utile' ou 'Inutile' avant de valider.")

        # Bouton pour réinitialiser le chat
        if st.button("Réinitialiser le Chat", type="primary"):
            st.session_state.messages = []
            st.session_state.feedback_data = {}  # 🔹 Réinitialiser les feedbacks aussi
            st.rerun()










    # def show_chats(self):
    #     """Display chat interface"""
    #     st.title("💬 Mes conversations")
        
    #     try:
    #         self._display_chat_interface()
    #     except Exception as e:
    #         st.error(f"Erreur lors de l'affichage du chat: {str(e)}")
    #         st.button("🔄 Réessayer", on_click=st.rerun)

    # def _display_chat_interface(self):
    #     """Handle chat interface display and interaction"""
    #     st.subheader(f"📝 {st.session_state.current_chat}")

    #     # Display chat history
    #     st.divider()
    #     for message in st.session_state.chat_history:
    #         with st.chat_message(message["role"]):
    #             st.markdown(message["content"])

    #     # Chat input
    #     if prompt := st.chat_input("Votre message..."):
    #         self._handle_chat_input(prompt)

    # def _handle_chat_input(self, prompt):
    #     """Process chat input and generate response"""
    #     with st.chat_message("user"):
    #         st.markdown(prompt)
    #     st.session_state.chat_history.append({"role": "user", "content": prompt})

    #     with st.chat_message("assistant"):
    #         with st.spinner("Réflexion en cours..."):
    #             try:
    #                 response = self.llm(
    #                     query=prompt,
    #                     history=st.session_state.chat_history
    #                 )
    #                 st.markdown(response)
    #                 st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
    #                 # self.db.add_query(
    #                 #     query=response,
    #                 #     username=st.session_state.username,
    #                 #     chat_title=st.session_state.current_chat
    #                 # )
    #             except Exception as e:
    #                 st.error(f"Erreur: {str(e)}")

    def show_stats(self):
        """Display statistics dashboard"""
        st.title("📊 Statistiques")
        df = self.db.ask_line_plot_user(st.session_state.username)
        st.write(df.head())
        st.line_chart(df["avg_latency"])
        
        # Display key metrics and activity chart
        # get_line_plot_user(st.session_state.username)
        # self._display_metrics()
        # self._display_activity_chart()

    def _display_metrics(self):
        """Display key metrics"""
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="Total conversations",
                value=len(st.session_state.chat_history),
                delta="↑ 2 cette semaine"
            )
        with col2:
            st.metric(
                label="Messages envoyés",
                value=sum(1 for msg in st.session_state.chat_history if msg["role"] == "user"),
                delta="↑ 5 aujourd'hui"
            )
        with col3:
            st.metric(
                label="Temps moyen de réponse",
                value="2.3s",
                delta="-0.5s"
            )

    def _display_activity_chart(self):
        """Display activity visualization"""
        st.subheader("Activité hebdomadaire")
        dates = pd.date_range(start='2024-03-01', end='2024-03-07')
        activity_data = pd.DataFrame({
            'date': dates,
            'messages': np.random.randint(5, 25, size=len(dates))
        })
        
        fig = px.line(
            activity_data, 
            x='date', 
            y='messages',
            title="Messages par jour"
        )
        st.plotly_chart(fig, use_container_width=True)

    def show_sidebar(self):
        """Display enhanced sidebar"""
        with st.sidebar:
            self._display_user_profile()
            self._display_chat_controls()
            self._display_recent_chats()
            self._display_navigation()

    def _display_user_profile(self):
        """Display user profile section in sidebar"""
        st.markdown(f"""
            <div style='
                background: linear-gradient(135deg, #4f46e5, #3b82f6);
                padding: 20px;
                border-radius: 10px;
                color: white;
                margin-bottom: 20px;
            '>
                <h2 style='margin:0'>👤 Mon Espace</h2>
                <p style='margin:5px 0 0 0; opacity:0.9'>
                    {st.session_state.username}
                </p>
            </div>
        """, unsafe_allow_html=True)

    def _display_chat_controls(self):
        """Display chat control buttons"""
        if st.button("💬 Continuer Conversation"): 
            st.session_state.page = "chats"
            st.rerun()

        if st.button("➕ Nouvelle conversation", use_container_width=True, type="primary"):
            st.session_state.current_chat = f"Chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.session_state.chat_history = []
            st.rerun()
        st.divider()

    def _display_recent_chats(self):
        """Display recent chats in sidebar"""
        st.subheader("🕒 Récents")
        recent_chats = self.db.get_chat_history(username=st.session_state.username)
        
        for chat in recent_chats:
            self._display_chat_item(chat)

    def _display_chat_item(self, chat):
        """Display individual chat item in sidebar"""
        with st.container():
            st.markdown(f"""
                <div style='
                    background: {'#eef2ff' if chat['title'] == st.session_state.current_chat else 'white'};
                    border: 1px solid #e5e7eb;
                    border-radius: 8px;
                    padding: 12px;
                    margin: 8px 0;
                    cursor: pointer;
                    transition: all 0.2s;
                '>
                    <div style='display:flex; justify-content:space-between; align-items:center'>
                        <h4 style='margin:0; color:#1f2937'>💭 {chat['title']}</h4>
                        <span style='color:#6b7280; font-size:0.8em'>
                            {chat['date'].strftime('%H:%M')}
                        </span>
                    </div>
                    <p style='margin:5px 0 0 0; font-size:0.8em; color:#6b7280'>
                        ✉️ {len(chat['messages'])} messages
                    </p>
                </div>
            """, unsafe_allow_html=True)

    def _display_navigation(self):
        """Display navigation buttons in sidebar"""
        st.divider()
        if st.button("📊 Statistiques", use_container_width=True):
            st.session_state.page = "stats"
            st.rerun()

        if st.button("🎓 Quiz", use_container_width=True):
            st.session_state.page = "quiz"
            st.rerun()

        if st.button("🚪 Déconnexion", use_container_width=True, type="secondary"):
            st.session_state.clear()
            st.rerun()

    def show(self) -> None:
        """Main entry point for displaying the dashboard."""
        try:
            self.show_sidebar()
            
            # Route to appropriate page
            pages = {
                "chats": self.show_chats,
                "stats": self.show_stats,
                "quiz": self.show_quiz
            }
            
            current_page = st.session_state.page
            if current_page in pages:
                pages[current_page]()
            else:
                # logger.warning(f"Invalid page requested: {current_page}")
                st.error("Page not found")
                st.session_state.page = "chats"
                st.rerun()
                
        except Exception as e:
            # logger.error(f"Dashboard display error: {e}")
            st.error("An error occurred. Please try again.")
            if st.button("🔄 Retry"):
                st.rerun()


def show():
    dashboard = UserDashboard()
    dashboard.show()

if __name__ == "__main__":
    show()