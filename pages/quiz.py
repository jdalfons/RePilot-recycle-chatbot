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

username = "user"  # 🔹 À remplacer par un vrai système d'authentification

# ✅ Cache pour éviter de recharger les mêmes questions en boucle
@st.cache_data(ttl=600)
def get_cached_quiz_questions(username: str):
    return db.get_quiz_questions(username=username, limit=5)

# ✅ Initialisation des données stockées en `session_state`
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

# ✅ **Affichage des résultats uniquement après validation**
if st.session_state.get("show_results", False):

    # ✅ Calcul du pourcentage de réussite
    total_questions = len(st.session_state.quiz_data)
    score_percentage = (st.session_state.score / total_questions) * 100

    # ✅ Définition du message de motivation selon le score
    if st.session_state.score == total_questions:
        message = "🏆 Parfait ! Tu es un véritable expert du tri et du recyclage ! 🌱♻️"
        color = "green"
    elif st.session_state.score >= total_questions * 0.6:
        message = "💪 Bien joué !Encore un petit effort pour être imbattable ! 🔥"
        color = "blue"
    else:
        message = "😕 Oups... Il va falloir réviser un peu ! Essaie encore. 🔄"
        color = "red"

    # ✅ Affichage du MODAL POPUP 🎉
    with st.popover("📊 Résultats du quiz", use_container_width=True):
        st.markdown(f"<h2 style='text-align: center; color: {color};'>Score : {st.session_state.score}/{total_questions} ({score_percentage:.1f}%)</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; font-size: 18px; font-weight: bold;'>{message}</p>", unsafe_allow_html=True)

        # ✅ **Ajout d'une barre de progression du score**
        st.progress(score_percentage / 100)

        # ✅ Affichage de chaque question avec le bon/mauvais choix
        for i, quiz in enumerate(st.session_state.quiz_data):
            question_key = f"question_{i}"
            selected_answer = st.session_state.answers.get(question_key, "")

            if selected_answer == quiz["correct_answer"]:
                st.success(f"✔️ **Question {i+1} : Bonne réponse ✅**")
            else:
                st.error(f"❌ **Question {i+1} : Mauvaise réponse ❌**\n👉 **La bonne réponse était** : {quiz['correct_answer']}")

        # ✅ **Bouton pour fermer le modal sans recharger la page**
        if st.button("❌ Fermer les résultats"):
            st.session_state.show_results = False

        # ✅ **Bouton pour relancer un nouveau quiz**
        if st.button("🔄 Recommencer le quiz"):
            del st.session_state.quiz_data
            del st.session_state.answers
            del st.session_state.validated
            del st.session_state.show_results
            st.rerun()
