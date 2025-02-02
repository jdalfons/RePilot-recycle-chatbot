import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from database.db_management import SQLDatabase
from datetime import datetime
from rag_simulation.rag_augmented import AugmentedRAG
from rag_simulation.corpus_ingestion import BDDChunks
from plots.plots import get_line_plot_user


from dotenv import find_dotenv, load_dotenv
from database.db_management import db
from utils import Config
import random



class UserDashboard:
    def __init__(self):
        self.db = SQLDatabase(db_name="poc_rag")
        self.llm = self._init_cached_llm()
        self.init_session_state()
        self.df = self.db.ask_line_plot_user(st.session_state.username)

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

        st.title(" Challenge du Tri & Recyclage !♻️")

        username = st.session_state.username # 🔹 Replace with actual authentication system

        # ✅ Cache function to prevent reloading the same questions repeatedly
        @st.cache_data(ttl=600)
        def get_cached_quiz_questions(username: str):
            return db.get_quiz_questions(username=username, limit=5)

        # ✅ Initialize session state variables
        if "quiz_data" not in st.session_state:
            st.session_state.quiz_data = get_cached_quiz_questions(username=username)
            st.session_state.answers = {}  # Store user's answers
            st.session_state.validated = False  # Indicates if the quiz has been validated
            st.session_state.show_results = False  # Display results after validation

        quiz_data = st.session_state.quiz_data  # Load cached questions

        # ✅ Check if questions are available
        if not quiz_data or "⚠️" in quiz_data[0].get("message", ""):
            st.warning("⚠️ Aucun historique de questions trouvé pour cet utilisateur.")
        else:
            for i, quiz in enumerate(quiz_data):
                question_key = f"question_{i}"

                st.write(f"### Question {i+1}: {quiz['question']}")
   
                # ✅ Shuffle answers and store them in session_state upon first display
                if question_key not in st.session_state:
                    answers = [quiz["correct_answer"]] + quiz["fake_answers"]
                    random.shuffle(answers)
                    st.session_state[question_key] = answers  # Save shuffled order

                # ✅ Load stored answers
                answers = st.session_state[question_key]

                # ✅ User answer selection
                selected_answer = st.radio(
                    f"Choisissez la bonne réponse :",
                    options=answers,
                    key=f"quiz_radio_{i}",
                    index=answers.index(st.session_state.answers.get(question_key, answers[0])) if question_key in st.session_state.answers else 0
                )

                # ✅ Store selected answer
                st.session_state.answers[question_key] = selected_answer

            # ✅ Single validation button for all answers
            if st.button("✅ Valider le quiz") and not st.session_state.validated:
                st.session_state.validated = True  # Mark as validated
                st.session_state.show_results = True  # Enable result display

                # ✅ Compute final score
                st.session_state.score = sum(
                    1 for i, quiz in enumerate(quiz_data)
                    if st.session_state.answers.get(f"question_{i}", "") == quiz["correct_answer"]
                )

        # ✅ **Show results only after validation**
        if st.session_state.get("show_results", False):

            # ✅ Compute success percentage
            total_questions = len(st.session_state.quiz_data)
            score_percentage = (st.session_state.score / total_questions) * 100

            # ✅ Define motivation message based on score
            if st.session_state.score == total_questions:
                message = "🏆 Parfait ! Tu es un véritable expert du tri et du recyclage ! 🌱♻️"
                color = "green"
            elif st.session_state.score >= total_questions * 0.6:
                message = "💪 Bien joué ! Encore un petit effort pour être imbattable ! 🔥"
                color = "blue"
            else:
                message = "😕 Oups... Il va falloir réviser un peu ! Essaie encore. 🔄"
                color = "red"

            # ✅ Display results in a **MODAL POPUP** 🎉
            with st.popover("📊 Résultats du quiz", use_container_width=True):
                st.markdown(f"<h2 style='text-align: center; color: {color};'>Score : {st.session_state.score}/{total_questions} ({score_percentage:.1f}%)</h2>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; font-size: 18px; font-weight: bold;'>{message}</p>", unsafe_allow_html=True)

                # ✅ **Add progress bar for score**
                st.progress(score_percentage / 100)

                # ✅ Display each question with correct/incorrect feedback
                for i, quiz in enumerate(st.session_state.quiz_data):
                    question_key = f"question_{i}"
                    selected_answer = st.session_state.answers.get(question_key, "")

                    if selected_answer == quiz["correct_answer"]:
                        st.success(f"✔️ **Question {i+1} : Bonne réponse ✅**")
                    else:
                        st.error(f"❌ **Question {i+1} : Mauvaise réponse ❌**\n👉 **La bonne réponse était** : {quiz['correct_answer']}")

                # ✅ **Button to close the modal without reloading**
                if st.button("❌ Fermer les résultats"):
                    st.session_state.show_results = False

                # ✅ **Button to restart a new quiz**
                if st.button("🔄 Recommencer le quiz"):
                    del st.session_state.quiz_data
                    del st.session_state.answers
                    del st.session_state.validated
                    del st.session_state.show_results
                    st.rerun()


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
    
        # Add custom CSS for container sizing
        st.markdown("""
            <style>
                .stPlotlyChart {
                    width: 70%;
                }
                .block-container {
                    padding-top: 1rem;
                    padding-bottom: 1rem;
                    max-width: 87%;
                }
                [data-testid="stMetricValue"] {
                    width: 70%;
                }
            </style>
        """, unsafe_allow_html=True)
        
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

        # On clear le cache pour relancer la fonction si la ville a changé
        if 'previous_city' not in st.session_state:
            st.session_state['previous_city'] = st.session_state["ville_choisi"]

        if st.session_state['previous_city'] != st.session_state["ville_choisi"]:
            st.cache_resource.clear()
            st.session_state['previous_city'] = st.session_state["ville_choisi"]



        col1, col2 = st.columns([1, 2])
        with col1:
            
            generation_model = st.selectbox(
                label="🤖 Choississez Assistant RePilot",
                # label="Choisissez votre modèle LLM",
                options=[
                    "ministral-8b-latest",
                    "ministral-3b-latest",
                    "codestral-latest",
                    "mistral-large-latest",
                ],
            )

        with col2:
            # role_prompt = st.text_area(
            #     label=config_chatbot.get('label'),
            #     value=config_chatbot.get('value'),
            # )

            # Options avancées
            st.write("🛠️ Options avancées")
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
        role_prompt = config.get_role_prompt()
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
        

        # initialisation de `response` et `query_id` pour éviter les erreurs
        historique = db.get_chat_history_user(st.session_state.username)
        st.session_state.messages = historique
        
        # Afficher l’historique des messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                #ameliorer  mettre une  barre deroulante pour les messages
                st.markdown(message["content"])
                # st.markdown(message["content"])

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
                username1=st.session_state.username
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
                        # success = db.save_feedback(
                        #     query_id=query_id,
                        #     username="user",
                        #     feedback=selected_feedback,
                        #     comment=comment
                        # )
                        success = True  
                        if success:
                            st.success(f"✅ Merci pour votre retour : {selected_feedback}")
                        else:
                            st.error("❌ Une erreur est survenue lors de l'enregistrement du feedback.")
                    else:
                        st.warning("⚠️ Merci de sélectionner 'Utile' ou 'Inutile' avant de valider.")

    def show_stats(self):
        """Display enhanced statistics dashboard"""

            # Add custom CSS for container sizing
        st.markdown("""
            <style>
                .stPlotlyChart {
                    width: 100%;
                }
                .block-container {
                    padding-top: 1rem;
                    padding-bottom: 1rem;
                    max-width: 100%;
                }
                [data-testid="stMetricValue"] {
                    width: 100%;
                }
            </style>
        """, unsafe_allow_html=True)
        # Filters
        with st.sidebar:
            st.markdown("## 🎯 Filters")
            time_period = st.selectbox("Time Period", 
                ["Last 24h", "Last Week", "Last Month", "All Time"])
            
            # Filter data based on selection
            now = pd.Timestamp.now()
            if time_period == "Last 24h":
                filtered_df = self.df[self.df['timestamp'] > (now - pd.Timedelta(days=1))]
            elif time_period == "Last Week":
                filtered_df = self.df[self.df['timestamp'] > (now - pd.Timedelta(days=7))]
            elif time_period == "Last Month":
                filtered_df = self.df[self.df['timestamp'] > (now - pd.Timedelta(days=30))]
            else:
                filtered_df = self.df

        # Metrics Grid
        st.markdown("## 📊 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = {
            "⚡ Latency": f"{filtered_df['avg_latency'].mean():.1f}ms",
            "💰 Cost": f"${filtered_df['avg_query_price'].sum():.4f}",
            "🔋 Energy": f"{filtered_df['avg_energy_usage'].sum():.4f}kWh",
            "🌍 GWP": f"{filtered_df['avg_gwp'].sum():.4f}kg"
        }
        
        for col, (title, value) in zip([col1, col2, col3, col4], metrics.items()):
            with col:
                st.markdown(f"""
                    <div style='
                        background: white;
                        padding: 1.5em;
                        border-radius: 10px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        text-align: center;
                    '>
                        <h4 style='margin:0; color:#6b7280'>{title}</h4>
                        <h2 style='margin:0.5em 0; color:#2563eb'>{value}</h2>
                    </div>
                """, unsafe_allow_html=True)

        # Charts
        st.markdown("### 📈 Analyse")
        chart_tabs = st.tabs(["Cout", "performance"])
        with chart_tabs[0]:
            col1, col2 , col3= st.columns(3)
            with col1:
                fig_cost = self.create_trend_chart(
                    filtered_df, 
                    'avg_query_price',
                    'Query Cost',
                    'Time',
                    'Cost'
                )
                st.plotly_chart(fig_cost, use_container_width=True)
            with col2:
                fig_energy = self.create_trend_chart(
                    filtered_df, 
                    'avg_energy_usage',
                    'Energy Usage',
                    'Time',
                    'kWh'
                )
                st.plotly_chart(fig_energy, use_container_width=True)
            with col3:
                fig_gwp = self.create_trend_chart(
                    filtered_df, 
                    'avg_gwp',
                    'Global Warming Potential',
                    'Time',
                    'kg CO₂e'
                )
                st.plotly_chart(fig_gwp, use_container_width=True)
        with chart_tabs[1]:
            col1, col2, col3 = st.columns(3)
            with col1:
                fig_latency = self.create_trend_chart(
                    filtered_df, 
                    'avg_latency',
                    'Response Latency',
                    'Time',
                    'Seconds'
                )
                st.plotly_chart(fig_latency, use_container_width=True)
            with col2:
                fig_tokens = self.create_trend_chart(
                    filtered_df, 
                    'avg_completion_tokens',
                    'Completion Tokens',
                    'Time',
                    'Tokens'
                )
                st.plotly_chart(fig_tokens, use_container_width=True)
            with col3:
                fig_tokens = self.create_trend_chart(
                    filtered_df, 
                    'avg_prompt_tokens',
                    'Prompt Tokens',
                    'Time',
                    'Tokens'
                )
                st.plotly_chart(fig_tokens, use_container_width=True)

    def create_trend_chart(self, df, metric, title, x_label, y_label):
        """Helper function to create trend charts"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df[metric],
            mode='lines',
            line=dict(color='#2563eb', width=2)
        ))
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            template='plotly_white',
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig
    
    def create_bar_chart(self, df, metric, title, x_label, y_label):
        """Helper function to create bar charts"""
        fig = px.bar(
            df,
            x='timestamp',
            y=metric,
            title=title,
            labels={'timestamp': x_label, metric: y_label},
            template='plotly_white'
        )
        return fig
    
    def _display_metrics_sidebar(self): 
        """Display key metrics in sidebar"""
        st.sidebar.subheader("📊 Statistiques")
        st.sidebar.markdown("Afficher les statistiques clés de l'assistant.")
        st.sidebar.write("Total conversations: 12")
        st.sidebar.write("Messages envoyés: 45")
        st.sidebar.write("Temps moyen de réponse: 2.3s")


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
            self._display_user_stats()
            self._display_navigation()

    def _display_user_profile(self):
        """Display user profile section in sidebar"""
        st.markdown(f"""
            <div style='
                background: linear-gradient(135deg, #4f46e5, #3b82f6);
                padding: 10px;
                border-radius: 5px;
                color: white;
                margin-bottom: 10px;
            '>
                <h2 style='margin:0'>👤 Mon Espace</h2>
                <p style='margin:5px 0 0 0; opacity:0.9'>
                    {st.session_state.username}
                </p>
            </div>
        """, unsafe_allow_html=True)

    def _display_chat_controls(self):
        """Display chat control buttons"""
        if st.button("💬   Continuer Conversation" ,use_container_width=True): 
            st.session_state.page = "chats"
            st.rerun()

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

    def _display_user_stats(self):
        """Display user statistics in a formatted container"""
        # Calculate statistics

        num_conversations = self.df['timestamp'].count()
        avg_latency = self.df['avg_latency'].mean()
        total_cost = self.df['avg_query_price'].sum()
        total_energy = self.df['avg_energy_usage'].sum()
        avg_gwp = self.df['avg_gwp'].mean()

        
        with st.container():
            st.markdown("""
                <style>
                    .stat-card {
                        transition: all 0.3s ease;
                    }
                    .stat-card:hover {
                        transform: translateY(-4px);
                        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                    }
                </style>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style='
                    background: linear-gradient(to bottom right, #ffffff, #f8fafc);
                    border: 1px solid #e2e8f0;
                    border-radius: 0px;
                    padding: 5px;
                    margin: 12px 0;
                    box-shadow: 5 2px 4px rgba(0, 0, 0, 0.05);
                '>
                    <h10 style='margin:0 0 10px 0; color:#334155; font-weight:400'>📊 User Statistics</h10>
                    <div style='display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:20px'>
                        <div class='stat-card' style='background:#f8fafc; padding:16px; border-radius:8px; border:1px solid #e2e8f0'>
                            <p style='margin:0; font-size:0.9em; color:#64748b'>💬 Conversations</p>
                            <p style='margin:4px 0; font-size:1.4em; color:#334155; font-weight:400'>{num_conversations}</p>
                            <p style='margin:0; font-size:0.9em; color:#64748b'>⚡ Avg. Latency</p>
                            <p style='margin:4px 0; font-size:1.4em; color:#334155; font-weight:400'>{avg_latency:.1f}ms</p>
                            <p style='margin:0; font-size:0.9em; color:#64748b'>⚡ Avg. Latency</p>
                            <p style='margin:4px 0; font-size:1.4em; color:#334155; font-weight:400'>${total_cost:.4f}</p>
                            <p style='margin:0; font-size:0.9em; color:#64748b'>🔋 Energy Used</p>
                            <p style='margin:4px 0; font-size:1.4em; color:#334155; font-weight:400'>{total_energy:.4f} kWh</p>
                            <p style='margin:0; font-size:0.9em; color:#64748b'>🌍 Average GWP</p>
                            <p style='margin:4px 0; font-size:1.4em; color:#334155; font-weight:400'>{avg_gwp:.4f} kg CO₂e</p>
                        </div>
                        

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
            st.error("An error occurred. Please try again.")
            if st.button("🔄 Retry"):
                st.rerun()
    def __call__(self, *args, **kwds):
        self.df = self.db.ask_line_plot_user(st.session_state.username)
        # self.show()



def show():
    dashboard = UserDashboard()
    dashboard.show()

if __name__ == "__main__":
    show()