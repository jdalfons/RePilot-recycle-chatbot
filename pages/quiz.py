import streamlit as st
import random
from database.db_management import db

# ✅ Configuration de la page
st.set_page_config(
    page_title="Quiz",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Quiz basé sur vos questions")

username = "user"  # 🔹 À remplacer par un système d'authentification réel

# ✅ Cache pour éviter de recharger les mêmes questions en boucle
@st.cache_data(ttl=600)
def get_cached_quiz_questions(username: str):
    return db.get_quiz_questions(username=username, limit=5)

# ✅ Stocker les données dans `st.session_state` pour éviter leur réinitialisation
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = get_cached_quiz_questions(username=username)
    st.session_state.answers = {}  # Stocke les réponses de l'utilisateur
    st.session_state.validated = False  # Indique si le quiz a été validé
    st.session_state.show_results = False  # Affiche les résultats après validation

quiz_data = st.session_state.quiz_data  # Charger les questions stockées

# ✅ Vérifier si des questions existent
if not quiz_data or "⚠️" in quiz_data[0].get("message", ""):
    st.warning("⚠️ Aucun historique de questions trouvé pour cet utilisateur.")
else:
    for i, quiz in enumerate(quiz_data):
        question_key = f"question_{i}"

        st.write(f"### Question {i+1}: {quiz['question']}")

        # ✅ Mélanger les réponses et stocker dans `st.session_state` au premier affichage
        if question_key not in st.session_state:
            answers = [quiz["correct_answer"]] + quiz["fake_answers"]
            random.shuffle(answers)
            st.session_state[question_key] = answers  # Sauvegarde de l'ordre mélangé

        # ✅ Charger les réponses stockées
        answers = st.session_state[question_key]

        # ✅ Sélection de la réponse utilisateur
        selected_answer = st.radio(
            f"Choisissez la bonne réponse :",
            options=answers,
            key=f"quiz_radio_{i}",
            index=answers.index(st.session_state.answers.get(question_key, answers[0])) if question_key in st.session_state.answers else 0
        )

        # ✅ Stocker la réponse choisie
        st.session_state.answers[question_key] = selected_answer

    # ✅ Bouton unique pour valider toutes les réponses
    if st.button("✅ Valider le quiz") and not st.session_state.validated:
        st.session_state.validated = True  # Marquer comme validé
        st.session_state.show_results = True  # Activer l'affichage des résultats

        # ✅ Calcul du score final
        st.session_state.score = sum(
            1 for i, quiz in enumerate(quiz_data)
            if st.session_state.answers.get(f"question_{i}", "") == quiz["correct_answer"]
        )

# ✅ Affichage des résultats après validation
if st.session_state.get("show_results", False):
    st.markdown("---")  # Ligne de séparation
    st.subheader(f"📊 Résultats du quiz : {st.session_state.score}/{len(quiz_data)}")

    for i, quiz in enumerate(quiz_data):
        question_key = f"question_{i}"
        selected_answer = st.session_state.answers.get(question_key, "")

        if selected_answer == quiz["correct_answer"]:
            st.success(f"✔️ Question {i+1} : Bonne réponse ✅")
        else:
            st.error(f"❌ Question {i+1} : Mauvaise réponse ❌\n👉 **La bonne réponse était** : {quiz['correct_answer']}")

    # ✅ Message de motivation selon le score
    if st.session_state.score == len(quiz_data):
        st.success("🏆 Félicitations, vous avez fait un sans-faute ! 🎉")
    elif st.session_state.score > len(quiz_data) // 2:
        st.info("💪 Bon travail ! Vous pouvez encore progresser.")
    else:
        st.warning("😕 Il y a encore du travail ! Entraînez-vous davantage.")

    # ✅ Bouton pour réinitialiser le quiz
    if st.button("🔄 Recommencer le quiz"):
        del st.session_state.quiz_data
        del st.session_state.answers
        del st.session_state.validated
        del st.session_state.show_results
        st.rerun()
