import streamlit as st
from database.db_management import db
import random

st.title("🧠 Quiz basé sur vos questions")

username = "user"  # À remplacer par un système d'authentification

# ✅ Stocker les questions dans `st.session_state` pour éviter leur réinitialisation
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = db.get_quiz_questions(username=username, limit=5)
    st.session_state.answers = {}  # Stocke les réponses sélectionnées par l'utilisateur
    st.session_state.validated = False  # Indique si le quiz a été validé
    st.session_state.show_modal = False  # Indique si le modal doit être affiché

quiz_data = st.session_state.quiz_data  # Charger les questions stockées

if not quiz_data or "⚠️" in quiz_data[0].get("message", ""):
    st.warning("⚠️ Aucun historique de questions trouvé pour cet utilisateur.")
else:
    for i, quiz in enumerate(quiz_data):
        question_key = f"question_{i}"  # Identifiant unique par question

        st.write(f"### Question {i+1}: {quiz['question']}")

        # Mélanger les réponses uniquement si ce n'est pas déjà fait
        if question_key not in st.session_state:
            answers = [quiz["correct_answer"]] + quiz["fake_answers"]
            random.shuffle(answers)
            st.session_state[question_key] = answers  # Sauvegarder l'ordre mélangé

        # Charger l'ordre des réponses stocké
        answers = st.session_state[question_key]

        # ✅ Stocker la réponse sélectionnée dans `st.session_state`
        selected_answer = st.radio(
            f"Choisissez la bonne réponse :",
            options=answers,
            key=f"quiz_radio_{i}",
        )

        # Stocker la réponse sélectionnée pour validation
        st.session_state.answers[question_key] = selected_answer

    # ✅ Bouton unique pour valider toutes les réponses
    if st.button("✅ Valider le quiz", key="validate_quiz") and not st.session_state.validated:
        st.session_state.validated = True  # Marquer le quiz comme validé
        st.session_state.show_modal = True  # Activer l'affichage du modal

        # ✅ Calculer le score
        st.session_state.score = sum(
            1 for i, quiz in enumerate(quiz_data)
            if st.session_state.answers.get(f"question_{i}", "") == quiz["correct_answer"]
        )

# ✅ Affichage des résultats dans un **MODAL**
if st.session_state.get("show_modal", False):
    with st.expander("📊 Résultats du quiz", expanded=True):
        st.subheader(f"🎯 Score final : {st.session_state.score}/{len(quiz_data)}")

        # ✅ Afficher chaque question avec l’état de la réponse
        for i, quiz in enumerate(quiz_data):
            question_key = f"question_{i}"
            selected_answer = st.session_state.answers.get(question_key, "")

            if selected_answer == quiz["correct_answer"]:
                st.success(f"✔️ Question {i+1} : Bonne réponse ✅")
            else:
                st.error(f"❌ Question {i+1} : Mauvaise réponse ❌\n👉 La bonne réponse était : {quiz['correct_answer']}")

        # ✅ Message de motivation selon le score
        if st.session_state.score == len(quiz_data):
            st.success("🏆 Félicitations, vous avez fait un sans-faute ! 🎉")
        elif st.session_state.score > len(quiz_data) // 2:
            st.info("💪 Bon travail ! Vous pouvez encore progresser.")
        else:
            st.warning("😕 Il y a encore du travail ! Entraînez-vous davantage.")

    # ✅ Bouton pour réinitialiser le quiz et relancer une nouvelle session
    if st.button("🔄 Recommencer le quiz"):
        del st.session_state.quiz_data
        del st.session_state.answers
        del st.session_state.validated
        del st.session_state.show_modal
        st.rerun()
