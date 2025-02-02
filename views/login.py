import streamlit as st
import hashlib
from pathlib import Path
from database.db_management import SQLDatabase

class LoginInterface:
    def __init__(self):
        self.db = SQLDatabase(db_name="poc_rag")
        self.load_css()
        self.init_session_state()
        
    def init_session_state(self):
        """Initialize all required session state variables"""
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False
        if "username" not in st.session_state:
            st.session_state.username = None 
        if "role" not in st.session_state:
            st.session_state.role = None
        if "tab" not in st.session_state:
            st.session_state.tab = "login"
            
    def load_css(self):
        st.markdown("""
        <style>
        :root {
            --primary: #4A90E2;
            --success: #2ECC71;
            --error: #E74C3C;
        }
        .title-container { text-align: center; margin-bottom: 2rem; }
        .stButton { display: flex; justify-content: flex-end; }
        .stButton > button {
            width: auto;
            min-width: 120px;
            background: var(--primary);
            padding: 12px 24px;
            border-radius: 8px;
        }
        </style>
        """, unsafe_allow_html=True)

    def validate_password(self, password: str) -> tuple[bool, str]:
        if len(password) < 6:
            return False, "Le mot de passe doit contenir au moins 6 caractères"
        if not any(c.isdigit() for c in password):
            return False, "Le mot de passe doit contenir au moins 1 chiffre"
        if not any(c.isupper() for c in password):
            return False, "Le mot de passe doit contenir au moins 1 majuscule"
        return True, "Mot de passe valide"

    def show_login_form(self):
        username = st.text_input("Nom d'utilisateur", key="login_user")
        password = st.text_input("Mot de passe", type="password", key="login_pass")
        
        if st.button("Se connecter", type="primary"):
            if not username or not password:
                st.toast("⚠️ Veuillez remplir tous les champs", icon="⚠️")
            else:
                with st.spinner("Authentification..."):
                    if self.db.verify_password(username, password):
                        st.toast("✅ Connexion réussie!", icon="✅")
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        try:
                            st.session_state.role = self.db.get_user_role(username)
                        except Exception as e:
                            st.error(f"Échec de la récupération du rôle: {e}")
                        st.rerun()
                    else:
                        st.toast("❌ Identifiants invalides", icon="❌")

    def show_register_form(self):
        username = st.text_input("Nom d'utilisateur", key="reg_user")
        password = st.text_input("Mot de passe", type="password", 
                               placeholder="Min. 6 caractères, 1 chiffre, 1 majuscule",
                               key="reg_pass")
        confirm_password = st.text_input("Confirmer le mot de passe", 
                                       type="password", key="reg_confirm")
        
        if st.button("Créer un compte", type="primary"):
            if not username or not password or not confirm_password:
                st.toast("⚠️ Veuillez remplir tous les champs", icon="⚠️")
            else:
                valid, message = self.validate_password(password)
                if not valid:
                    st.toast(f"⚠️ {message}", icon="⚠️")
                elif password != confirm_password:
                    st.toast("❌ Les mots de passe ne correspondent pas", icon="❌")
                elif self.db.verifier_si_utilisateur(username):
                    st.toast("⚠️ Ce nom d'utilisateur existe déjà", icon="⚠️")
                else:
                    if self.db.create_user(username, password):
                        st.toast("✅ Compte créé! Veuillez vous connecter.", icon="✅")
                        st.session_state.tab = "login"

    def show(self):
        """Display login interface and handle authentication"""
        if not st.session_state.authenticated:
            col1, col2, col3 = st.columns([4,2,4])
            with col2:
                image_path = Path("assets/image.png")
                if image_path.exists():
                    st.image(str(image_path), width=100)
            
            st.markdown("<div class='title-container'><h1>🚀 Bienvenue!</h1></div>", 
                    unsafe_allow_html=True)
            
            tab1, tab2 = st.tabs(["🔑 Connexion", "📝 Inscription"])
            with tab1: self.show_login_form()
            with tab2: self.show_register_form()

        # Safe access to session state with defaults
        return (
            st.session_state.get("authenticated", False),
            st.session_state.get("username", None)
        )

def show():
    interface = LoginInterface()
    return interface.show()

if __name__ == "__main__":
    show()