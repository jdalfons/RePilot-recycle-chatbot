from typing import Dict, List, Optional, Any
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from database.db_management import SQLDatabase
import logging
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


class AdminDashboard:
    def __init__(self) -> None:
        try:
            self.db = SQLDatabase(db_name="poc_rag")
            self.init_session_state()
            logger.info("Admin dashboard initialized")
        except Exception as e:
            logger.error(f"Init error: {e}")
            st.error("Dashboard initialization failed")
            raise

    def init_session_state(self) -> None:
        defaults = {
            "admin_page": "overview",
            "selected_user": None,
            "date_range": "7d",
            "filter_status": "all",
            "view_mode": "grid",
            "selected_metrics": ["cpu", "memory", "response_time"],
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def show(self) -> None:
        try:
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
            self.show_sidebar()

            pages = {
                "overview": self.show_overview,
                "users": self.show_users,
                # "performance": self.show_performance_quizz,
                "logout": self.logout,
            }

            if st.session_state.admin_page in pages:
                pages[st.session_state.admin_page]()
            else:
                st.error("Page not found")
                st.session_state.admin_page = "overview"
                st.rerun()
        except Exception as e:
            logger.error(f"Display error: {e}")
            st.error("Error loading dashboard")

    def show_sidebar(self) -> None:
        with st.sidebar:
            st.title("🎛️ Admin Panel")

            # Navigation
            pages = {
                "📊 Overview": "overview",
                "👥 Users": "users",
                # "📈 Performance Quizz": "performance",
                "🔒 Logout": "logout",
            }

            st.divider()
            for label, page in pages.items():
                if st.button(
                    label,
                    use_container_width=True,
                    type=(
                        "primary"
                        if st.session_state.admin_page == page
                        else "secondary"
                    ),
                ):
                    st.session_state.admin_page = page
                    st.rerun()
    
    def logout(self) -> None:
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.role = None
        st.session_state.current_page = "login"
        st.session_state.admin_page = "overview"
        st.rerun()

    ##################OVERVIEW PAGE##################
    def show_overview(self) -> None:
        st.title("📊 System Overview")

        stats = self.get_usage_statistics()

        # First row - User Activity Metrics
        users_col, chats_col, queries_col = st.columns(3)
        with users_col:
            st.metric("👥 Total Users", stats.get("total_users", 0))
        with chats_col:
            st.metric("💬 Active Chats", stats.get("total_chats", 0))
        with queries_col:
            queries = stats.get("total_queries", 0)
            st.metric("📝 Total Queries", f"{queries:,}")

        # Second row - Performance Metrics
        latency_col, safety_col, model_col = st.columns(3)
        with latency_col:
            latency = round(stats.get("avg_latency", 0), 2)
            st.metric("⚡ Response Time", f"{latency}ms")
        with safety_col:
            safe = round(stats.get("safe_queries_percentage", 0), 2)
            st.metric("🛡️ Safe Queries", f"{safe}%")
        with model_col:
            most_used_model = stats.get("most_used_model", "N/A")
            st.metric("🧠 Most Used Model", most_used_model)

        # Third row - Environmental & Cost Metrics
        energy_col, impact_col, cost_col = st.columns(3)
        with energy_col:
            energy = round(stats.get("total_energy", 0) * 1_000, 2)
            st.metric("⚡ Energy Usage", f"{energy:,.2f} mWh")
        with impact_col:
            gwp = round(stats.get("total_gwp", 0) * 1_000, 2)
            st.metric("🌍 GWP Impact", f"{gwp:,.2f} gCO2eq")
        with cost_col:
            cost = round(stats.get("total_cost", 0), 2)
            st.metric("💰 Total Cost", f"${cost:,.2f}")

        # Charts
        self._show_activity_charts()

    ###########################="USERS PAGE=###################

    def show_users(self) -> None:
        st.title("👥 User Management")
 
        users = self.db.get_usernames()

        # Show the list of users
        overview_tab, user_details_tab = st.tabs(["📊 Overview", "👤 User Details"])
    
        with overview_tab:        
            st.subheader("Vue des Utilisateurs qui consomment le plus")

            # Plot Users that consume the most
            metric = st.selectbox(
                "Select metric to analyze",
                options=["Money Spent", "Environmental Impact", "Latency"],
                index=0,
                help="Choose the metric to display the top 5 users.",
            )

            # Get the top 5 users by the selected metric
            top_users = self.db.get_top_users_by_metric(metric)

            if top_users:
                # Colors palette
                colors = px.colors.qualitative.Pastel

                # 🎭 Affichage de la section avec un meilleur titre
                st.subheader(f"🏆 Top 5 Users by {metric}")

                # DataFrame
                df = pd.DataFrame(top_users, columns=["Username", metric])

                # Dynamically set the units and labels based on the selected metric
                if metric == "Money Spent":
                    unit = "$"
                    yaxis_title = "Money Spent ($)"
                    texttemplate = "%{text:.2f} $"
                elif metric == "Environmental Impact":
                    unit = "kgCO2eq"
                    yaxis_title = "Environmental Impact (kgCO2eq)"
                    texttemplate = "%{text:.2f} kgCO2eq"
                else:  # For "Latency"
                    unit = "ms"
                    yaxis_title = "Latency (ms)"
                    texttemplate = "%{text:.2f} ms"

                # Color by user
                fig = px.bar(
                    df,
                    x="Username",
                    y=metric,
                    title=f"Top Users by {metric}",
                    color="Username",
                    color_discrete_sequence=colors,
                    text=metric,
                    template="plotly_white",
                )

                # Update the chart with the corresponding units and style
                fig.update_traces(
                    texttemplate=texttemplate, textposition="outside", marker_line_width=1.5
                )
                fig.update_layout(
                    yaxis_title=yaxis_title,
                    xaxis_title="User",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=40, r=40, t=40, b=40),
                )

                # Show the data frame with formatted values
                st.dataframe(
                    df.style.format({metric: f"{{:.2f}} {unit}"}), use_container_width=True
                )

                # Display the plot
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No users found or no data available for the selected metric.")
            if not users:
                st.info("No users found.")
                return

        with user_details_tab:
            # Vue of specific user
            st.subheader("👤 Vue d'un Utilisateur spécifique")

            # select user
            selected_user = st.selectbox("Sélectionnez un utilisateur", users)

            # Get users' general info
            user_info = self.db.get_user_details(selected_user)
            user_stats = self.db.get_user_statistics(selected_user)
            user_feedback = self.db.get_user_feedback(selected_user)
            user_quiz = self.db.get_user_quiz_responses(selected_user)

            # user's general info
            st.subheader("📋 Informations de base")
            st.write(f"**Nom d'utilisateur :** {user_info['username']}")
            st.write(f"**Rôle :** {user_info['role']}")
            st.write(f"**Date de création :** {user_info['created_at']}")
            st.write(
                f"**Statut :** {'✅ Actif' if user_info['is_active'] else '❌ Inactif'}"
            )

            # Statistics of the user
            st.subheader("📊 Statistiques d'utilisation")
            stats_df = pd.DataFrame([user_stats])
            st.dataframe(
                stats_df.style.format(
                    {
                        "Money Spent": "{:.2f} $",
                        "Environmental Impact": "{:.2f} kgCO2eq",
                        "Latency": "{:.2f} ms",
                    }
                )
            )

            # Feedbacks of the user
            st.subheader("📝 Historique des feedbacks")
            if not user_feedback:
                st.write("Aucun feedback trouvé")
            else:
                feedback_df = pd.DataFrame(
                    user_feedback, columns=["Feedback", "Commentaire", "Date"]
                )
                st.dataframe(feedback_df)

            # Users' quiz history
            st.subheader("🎯 Historique des quiz")
            if not user_quiz:
                st.write("Aucune réponse aux quiz trouvée")
            else:
                quiz_df = pd.DataFrame(
                    user_quiz,
                    columns=["Question", "Réponse", "Bonne réponse", "Statut", "Date"],
                )
                quiz_df["Statut"] = quiz_df["Statut"].apply(
                    lambda x: "✅ Correct" if x else "❌ Incorrect"
                )
                st.dataframe(quiz_df)

            # Delete user
            st.subheader("🗑️ Supprimer l'utilisateur")
            if st.button(f"Supprimer {selected_user}", type="primary"):
                self.delete_user_and_data(selected_user)
                st.success(f"Utilisateur {selected_user} supprimé avec succès.")

    
    
    
     
    # Fonctions utils  02/02
    # Overview Page ============================================
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Retrieve usage statistics"""
        try:
            stats = self.db.get_usage_statistics()
            return stats
        except Exception as e:
            st.error(f"Error retrieving usage statistics: {e}")
            logger.error(f"Error retrieving usage statistics: {e}")
            return {}

    def _show_activity_charts(self) -> None:
        st.subheader("System Activity")

        # Sélection de la plage de dates
        start_date = st.date_input("Start Date", value=pd.to_datetime("2025-01-28"))
        end_date = st.date_input("End Date", value=pd.to_datetime("2025-02-03"))

        # Récupérer les données de la BDD
        activity_data = self.db.get_activity_data(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )
        # print(activity_data)

        # Vérifier si des données sont disponibles
        if not activity_data:
            st.warning("Aucune donnée trouvée pour cette période.")
            return

        # Conversion des données en DataFrame
        data = pd.DataFrame(activity_data, columns=["date", "users", "queries"])
        data["date"] = pd.to_datetime(data["date"])

        # Créer le graphique
        fig = px.line(
            data,
            x="date",
            y=["users", "queries"],
            labels={"value": "Count", "date": "Date", "variable": "Metric"},
            title="User Activity",
        )

        # Afficher le graphique dans Streamlit
        st.plotly_chart(fig, use_container_width=True)

    # Users Page ============================================
    def delete_user_and_data(self, username: str) -> None:
        """Delete a user and all associated data"""
        try:
            if self.db.delete_user_and_data(username):
                st.success(
                    f"User '{username}' and all associated data have been deleted."
                )
            else:
                st.error(f"Failed to delete user '{username}'.")
        except Exception as e:
            st.error(f"Error deleting user: {e}")
            logger.error(f"Error deleting user {username}: {e}")


def show():
    interface = AdminDashboard()
    return interface.show()


if __name__ == "__main__":
    show()
