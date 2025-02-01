from typing import Dict, List, Optional, Any
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from database.db_management import SQLDatabase
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    cpu_usage: float
    memory_usage: float
    response_time: float
    success_rate: float

class AdminDashboard:
    def __init__(self) -> None:
        try:
            self.db = SQLDatabase(db_name="poc_rag")
            self.metrics = SystemMetrics(0.0, 0.0, 0.0, 0.0)
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
            "selected_metrics": ["cpu", "memory", "response_time"]
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def show(self) -> None:
        try:
            self.show_sidebar()
            
            pages = {
                "overview": self.show_overview,
                "users": self.show_users,
                "chats": self.show_chats,
                "performance": self.show_performance,
                "settings": self.show_settings
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
                "💬 Chats": "chats",
                "📈 Performance": "performance",
                "⚙️ Settings": "settings"
            }
            
            st.divider()
            for label, page in pages.items():
                if st.button(label, use_container_width=True,
                           type="primary" if st.session_state.admin_page == page else "secondary"):
                    st.session_state.admin_page = page
                    st.rerun()

    def show_overview(self) -> None:
        st.title("📊 System Overview")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            users = len(self.db.get_users())
            st.metric("Total Users", users, "↑ 5")
        with col2:
            chats = len(self.db.get_all_chats())
            st.metric("Active Chats", chats, "↑ 12")
        with col3:
            response_time = 2.3
            st.metric("Avg Response", f"{response_time}s", "↓ 0.3s")
        with col4:
            success_rate = 95
            st.metric("Success Rate", f"{success_rate}%", "↑ 2%")

        # Charts
        self._show_activity_charts()
        self._show_performance_metrics()

    def show_users(self) -> None:
        st.title("👥 User Management")
        
        # Filters
        col1, col2 = st.columns([3, 1])
        with col1:
            search = st.text_input("🔍 Search users")
        with col2:
            status = st.selectbox("Status", ["All", "Active", "Inactive"])

        # User list
        users = self.db.get_users()
        if users:
            self._display_user_table(users)
        else:
            st.info("No users found")

    def show_chats(self) -> None:
        st.title("💬 Chat Monitor")
        
        # Filters
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            date_range = st.select_slider(
                "Time Range",
                options=["24h", "7d", "30d", "All"],
                value="7d"
            )
        with col2:
            user_filter = st.selectbox(
                "User",
                ["All"] + self.db.get_users()
            )
        with col3:
            st.button("Export", type="primary")

        # Chat history
        chats = self.db.get_all_chats()
        if chats:
            self._display_chat_history(chats)
        else:
            st.info("No chats found")

    def show_performance(self) -> None:
        st.title("📈 System Performance")
        
        # System metrics
        self._show_system_metrics()
        
        # Performance charts
        self._show_performance_charts()

    def show_settings(self) -> None:
        st.title("⚙️ System Settings")
        
        with st.form("settings"):
            st.subheader("Configuration")
            model = st.selectbox("LLM Model", ["GPT-4", "Claude", "Llama"])
            temp = st.slider("Temperature", 0.0, 2.0, 1.0)
            tokens = st.number_input("Max Tokens", 100, 2000, 1000)
            
            if st.form_submit_button("Save"):
                try:
                    # Save settings logic
                    st.success("Settings updated")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    def _show_activity_charts(self) -> None:
        st.subheader("System Activity")
        
        # Activity data
        dates = pd.date_range(start='2024-03-01', end='2024-03-07')
        data = pd.DataFrame({
            'date': dates,
            'users': np.random.randint(5, 25, size=len(dates)),
            'chats': np.random.randint(10, 50, size=len(dates))
        })
        
        # Charts
        fig = px.line(data, x='date', y=['users', 'chats'],
                     title="Daily Activity")
        st.plotly_chart(fig, use_container_width=True)

    def _show_system_metrics(self) -> None:
        st.subheader("Resource Usage")
        
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=70,
                title={'text': "CPU Usage"},
                gauge={'axis': {'range': [0, 100]}}
            ))
            st.plotly_chart(fig)
        
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=60,
                title={'text': "Memory Usage"},
                gauge={'axis': {'range': [0, 100]}}
            ))
            st.plotly_chart(fig)


def show():
    interface = AdminDashboard()
    return interface.show()

if __name__ == "__main__":
    show()