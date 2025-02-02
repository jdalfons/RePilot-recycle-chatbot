import streamlit as st
from database.db_management import SQLDatabase
from typing import List, Dict
import time

class ChatbotInterface:
    def __init__(self):
        self.db = SQLDatabase(db_name="poc_rag")
        
    def init_chat_state(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "chat_title" not in st.session_state:
            st.session_state.chat_title = f"Chat_{int(time.time())}"

    def display_chat_history(self):
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def handle_input(self, llm):
        if prompt := st.chat_input("Votre message..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Réflexion en cours..."):
                    response = llm(
                        query=prompt,
                        history=st.session_state.messages
                    )
                st.markdown(response)
                
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Save to database
            self.db.add_query(
                query=response,
                chat_title=st.session_state.chat_title,
                username=st.session_state.username
            )

    def show_chat_management(self):
        with st.sidebar:
            st.title("💬 Historique")
            
            if st.button("📝 Nouvelle conversation"):
                st.session_state.messages = []
                st.session_state.chat_title = f"Chat_{int(time.time())}"
                st.rerun()
                
            st.divider()
            
            # Load chat history from database
            chats = self.db.get_chat_history(st.session_state.username)
            for chat in chats:
                if st.button(f"🗨️ {chat['title']}", key=chat['chat_title']):
                    st.session_state.chat_title = chat['chat_title']
                    st.session_state.messages = chat['messages']
                    st.rerun()

def show():
    if not st.session_state.get("authenticated"):
        st.warning("Veuillez vous connecter pour accéder au chat")
        st.stop()
        
    interface = ChatbotInterface()
    interface.init_chat_state()
    
    st.title("🤖 Assistant RePilot")
    
    # Chat interface
    interface.show_chat_management()
    interface.display_chat_history()
    interface.handle_input(llm)

if __name__ == "__main__":
    show()