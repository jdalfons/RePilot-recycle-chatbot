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


    # def initialiser_message(self):
    #     """Initialise le message de bienvenue"""
    #     df  = db.get_chat_history_user(st.session_state.username)
    #     if len(df) == 0:
    #         return "Bonjour, comment puis-je vous aider ?"


    def  show_chats(self ):
        # st.title("💬 Mes conversations")
        # Charger la configuration

        # @st.cache_resource
        # def instantiate_bdd() -> BDDChunks:
        #     """Initialise la base ChromaDB et évite de la recharger plusieurs fois."""
        #     bdd = BDDChunks(embedding_model="paraphrase-multilingual-MiniLM-L12-v2")
        #     bdd()
        #     return bdd


        # Add custom CSS for container sizing
        st.markdown("""
            <style>
                .stPlotlyChart {
                    width: 80%;
                }
                .block-container {
                    padding-top: 1rem;
                    padding-bottom: 1rem;
                    max-width: 80%;
                }
                [data-testid="stMetricValue"] {
                    width: 80%;
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
        

        # initialisation de `response` et `query_id` pour éviter les erreurs
        historique = db.get_chat_history_user(st.session_state.username)
        st.session_state.messages = historique
        # st.write(historique[:5])
        # st.write(st.session_state.messages[:5])
    
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
            "💰 Cost": f"${filtered_df['avg_query_price'].mean():.4f}",
            "🔋 Energy": f"{filtered_df['avg_energy_usage'].mean():.4f}kWh",
            "🌍 GWP": f"{filtered_df['avg_gwp'].mean():.4f}kg"
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



        # chart_tabs = st.tabs(["Latency", "Cost", "Energy"])
        
        # with chart_tabs[0]:
        #     fig_latency = self.create_trend_chart(
        #         filtered_df, 
        #         'avg_latency',
        #         'Response Latency',
        #         'Time',
        #         'Seconds'
        #     )

            # st.plotly_chart(fig_latency, use_container_width=True)
            # st.bar_chart(filtered_df.groupby('timestamp')['avg_latency'].mean())

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

    # def show_stats(self):
    #     """Display statistics dashboard"""
    #     # Title Section
    #     st.title("📊 Analytics Dashboard")
        
    #     # Date Filter
    #     col1, col2 = st.columns(2)
    #     with col1:
    #         start_date = st.date_input("Start Date", value=self.df['timestamp'].min())
    #     with col2:
    #         end_date = st.date_input("End Date", value=self.df['timestamp'].max())
        
    #     # Filter data
    #     mask = (self.df['timestamp'].dt.date >= start_date) & (self.df['timestamp'].dt.date <= end_date)
    #     filtered_df = self.df[mask]
        
    #     # Key Metrics
    #     metrics_cols = st.columns(4)
    #     with metrics_cols[0]:
    #         st.metric("Avg Latency", f"{filtered_df['avg_latency'].mean():.2f}s")
    #     with metrics_cols[1]:
    #         st.metric("Total Queries", len(filtered_df))
    #     with metrics_cols[2]:
    #         st.metric("Avg Energy", f"{filtered_df['avg_energy_usage'].mean():.2f}kWh")
    #     with metrics_cols[3]:
    #         st.metric("Avg Cost", f"${filtered_df['avg_query_price'].mean():.3f}")
        
    #     # Charts
    #     st.subheader("Trends")
    #     tabs = st.tabs(["Latency", "Tokens", "Energy", "Cost"])
        
    #     with tabs[0]:
    #         st.line_chart(filtered_df.groupby('timestamp')['avg_latency'].mean())
        
    #     with tabs[1]:
    #         token_df = filtered_df[['avg_completion_tokens', 'avg_prompt_tokens']]
    #         st.line_chart(token_df)
        
    #     with tabs[2]:
    #         st.line_chart(filtered_df.groupby('timestamp')['avg_energy_usage'].mean())
        
    #     with tabs[3]:
    #         st.line_chart(filtered_df.groupby('timestamp')['avg_query_price'].mean())
        
    #     # Summary Statistics
    #     st.subheader("Summary Statistics")
    #     st.dataframe(filtered_df.describe())
        


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

    # def _display_user_stats(self):
    #     """Display user statistics in a formatted container"""
    #     # Calculate statistics
    #     num_conversations = self.df['timestamp'].count()
    #     avg_latency = self.df['avg_latency'].mean()
    #     total_cost = self.df['avg_price'].sum()
    #     total_energy = self.df['avg_energy_usage'].sum()
    #     avg_gwp = self.df['avg_gwp'].mean()
        
    #     with st.container():
    #         st.markdown(f"""
    #             <div style='
    #                 background: linear-gradient(to bottom right, #ffffff, #f8fafc);
    #                 border: 1px solid #e2e8f0;
    #                 border-radius: 12px;
    #                 padding: 24px;
    #                 margin: 16px 0;
    #                 box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    #                 transition: transform 0.2s ease;
    #             '>
    #                 <h3 style='margin:0 0 20px 0; color:#334155; font-weight:600'>📊 User Statistics</h3>
    #                 <div style='display:grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap:20px'>
    #                     <div class='stat-card' style='
    #                         background: #f8fafc;
    #                         padding: 16px;
    #                         border-radius: 8px;
    #                         border: 1px solid #e2e8f0;
    #                         transition: transform 0.2s ease;
    #                         &:hover { transform: translateY(-2px); }
    #                     '>
    #                         <p style='margin:0; font-size:0.9em; color:#64748b'>💬 Conversations</p>
    #                         <p style='margin:8px 0; font-size:1.4em; color:#334155; font-weight:600'>{num_conversations}</p>
    #                     </div>
    #                     <div class='stat-card' style='background:#f8fafc; padding:16px; border-radius:8px; border:1px solid #e2e8f0'>
    #                         <p style='margin:0; font-size:0.9em; color:#64748b'>⚡ Avg. Latency</p>
    #                         <p style='margin:8px 0; font-size:1.4em; color:#334155; font-weight:600'>{avg_latency:.2f}s</p>
    #                     </div>
    #                     <div class='stat-card' style='background:#f8fafc; padding:16px; border-radius:8px; border:1px solid #e2e8f0'>
    #                         <p style='margin:0; font-size:0.9em; color:#64748b'>💰 Total Cost</p>
    #                         <p style='margin:8px 0; font-size:1.4em; color:#334155; font-weight:600'>${total_cost:.2f}</p>
    #                     </div>
    #                     <div class='stat-card' style='background:#f8fafc; padding:16px; border-radius:8px; border:1px solid #e2e8f0'>
    #                         <p style='margin:0; font-size:0.9em; color:#64748b'>🔋 Energy Used</p>
    #                         <p style='margin:8px 0; font-size:1.4em; color:#334155; font-weight:600'>{total_energy:.2f} kWh</p>
    #                     </div>
    #                     <div class='stat-card' style='background:#f8fafc; padding:16px; border-radius:8px; border:1px solid #e2e8f0'>
    #                         <p style='margin:0; font-size:0.9em; color:#64748b'>🌍 Average GWP</p>
    #                         <p style='margin:8px 0; font-size:1.4em; color:#334155; font-weight:600'>{avg_gwp:.2f} kg CO₂e</p>
    #                     </div>
    #                 </div>
    #             </div>
    #         """, unsafe_allow_html=True)

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
    def __call__(self, *args, **kwds):
        self.df = self.db.ask_line_plot_user(st.session_state.username)
        # self.show()



def show():
    dashboard = UserDashboard()
    dashboard.show()

if __name__ == "__main__":
    show()