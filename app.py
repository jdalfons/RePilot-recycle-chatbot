import streamlit as st
from pathlib import Path
from database.db_management import SQLDatabase
import os

from views import admin_dashboard, login, user_dashboard
from rag_simulation.rag_augmented import AugmentedRAG


    # """Configure page settings"""
    # st.set_page_config(
    #     page_title="RePilot Chatbot",
    #     page_icon="🤖",
    #     layout="centered",
    #     initial_sidebar_state="expanded"
    # )

 
class MainApp:
    def __init__(self):
        self.db = SQLDatabase(db_name="poc_rag")
        self.pages = {
            "login": login,
            "admin_dashboard": admin_dashboard,
            "user_dashboard": user_dashboard,
        }
        self.init_session_state()
        
    def init_session_state(self):
        """Initialize session state variables"""
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False
        if "username" not in st.session_state:
            st.session_state.username = None
        if "role" not in st.session_state:
            st.session_state.role = None
        if "current_page" not in st.session_state:
            st.session_state.current_page = "login"

    def setup_page(self):
        """Configure page settings"""
        st.set_page_config(
            page_title="RePilot Chatbot",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def show_navigation(self):
        """Display navigation sidebar"""
        with st.sidebar:
            # pass
            # st.title("Navigation")
            
            if st.session_state.role == "admin":
                st.session_state.current_page =    "admin_dashboard"
            else:
                st.session_state.current_page = "user_dashboard"
                
            # page = st.radio("Aller à", pages)

            # st.session_state.current_page = page.lower().replace(" ", "_")
            
            # if st.button("Déconnexion"):
            #     st.session_state.authenticated = False
            #     st.session_state.username = None
            #     st.session_state.role = None
            #     st.rerun()

    def check_auth(self):
        """Verify authentication status"""
        if not st.session_state.authenticated:
            st.session_state.current_page = "login"
            return False
        return True

    def show_current_page(self):
        """Display current page content"""
        if st.session_state.current_page in self.pages:
            self.pages[st.session_state.current_page].show()
        else:
            st.error(f"Page {st.session_state.current_page} introuvable")

    def run(self):
        """Main application loop"""
        # self.setup_page()
        
        if not self.check_auth():
            self.pages["login"].show()
        else:
            self.show_navigation()
            self.show_current_page()

def main():


    app = MainApp()
    app.run()

if __name__ == "__main__":
    main()
