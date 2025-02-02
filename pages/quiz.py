import streamlit as st
import random
from database.db_management import db

# ✅ Page configuration
st.set_page_config(
    page_title="Quiz",
    page_icon="🧠",
    layout="wide",
)

st.title(" Challenge du Tri & Recyclage !♻️")

username = "user"  # 🔹 Replace with actual authentication system

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
