import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# -------------------- PAGE CONFIG -------------------- #
st.set_page_config(page_title="Study Habit Analyzer", page_icon="🎓", layout="centered")

# -------------------- PATHS -------------------- #
MODEL_DIR = Path("training/model")
DATASET_PATH = Path("dataset/courses.csv")

# -------------------- LOAD MODEL FILES -------------------- #
try:
    model = joblib.load(MODEL_DIR / "model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    course_enc = joblib.load(MODEL_DIR / "course_encoder.pkl")
    interest_enc_path = MODEL_DIR / "interest_encoder.pkl"
    interest_enc = joblib.load(interest_enc_path) if interest_enc_path.exists() else None
except Exception as e:
    st.error("❌ Some model files are missing in 'training/model'. Please make sure all .pkl files exist.")
    st.stop()

# -------------------- LOAD COURSE DATA -------------------- #
try:
    courses_df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    st.warning("⚠️ No dataset found. Please add 'courses.csv' in the 'dataset/' folder.")
    courses_df = pd.DataFrame(columns=["subject", "level", "name", "slug"])

# -------------------- UI SECTION -------------------- #
st.title("🎓 Study Habit Analyzer & Subject Recommender")
st.markdown("Analyze your study habits and get personalized subject recommendations!")

# -------------------- FORM INPUTS -------------------- #
with st.form("habit_form"):
    study_hours = st.slider("📚 Study Hours per Day", 0, 16, 4)
    attendance = st.slider("🏫 Attendance (%)", 0, 100, 85)
    test_score = st.slider("🧾 Average Test Score (%)", 0, 100, 70)
    focus_duration = st.slider("🕒 Focus Duration (minutes)", 5, 240, 45)
    distraction = st.slider("⚡ Distraction Level (1 = Low, 10 = High)", 1, 10, 5)
    
    # 10th Passed checkbox
    tenth_passed = st.checkbox("✅ 10th Passed?")
    
    # Stream selection if 10th passed
    stream = None
    if tenth_passed:
        stream = st.selectbox("📘 Select Your Preferred Stream", ["Science", "Commerce", "Arts"])
    
    interest_area = None
    if interest_enc is not None:
        interest_area = st.selectbox("💡 Your Main Interest Area", list(interest_enc.classes_))
    
    submitted = st.form_submit_button("🔍 Analyze My Habits")

# -------------------- ON SUBMIT -------------------- #
if submitted:
    # --- MODEL PREDICTION ---
    features = [study_hours, attendance, test_score, focus_duration, distraction]
    if interest_enc:
        if interest_area is None:
            st.warning("Please select your interest area.")
            st.stop()
        features.append(int(interest_enc.transform([interest_area])[0]))

    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    course_label = course_enc.inverse_transform([pred])[0]
    
    st.success(f"🎯 Recommended Course Level: **{course_label}**")

    # --- PERFORMANCE SCORECARD ---
    st.markdown("### 📊 Your Performance Scorecard")
    focus_score = round(min(focus_duration / 240 * 10, 10), 1)
    consistency_score = round(min(attendance / 10, 10), 1)
    test_prep_score = round(min(test_score / 10, 10), 1)

    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 Focus", f"{focus_score}/10")
    col2.metric("📅 Consistency", f"{consistency_score}/10")
    col3.metric("🧠 Test Preparation", f"{test_prep_score}/10")

    overall = round((focus_score + consistency_score + test_prep_score) / 3, 1)
    if overall >= 8:
        st.success(f"🌟 Overall Score: {overall}/10 — Excellent Performance!")
    elif overall >= 5:
        st.info(f"👍 Overall Score: {overall}/10 — Good, but can improve.")
    else:
        st.warning(f"⚠️ Overall Score: {overall}/10 — Needs improvement in study habits.")

    # --- STUDY IMPROVEMENT ADVICE ---
    st.markdown("### 🧠 Personalized Study Advice")
    tips = []
    if study_hours < 3:
        tips.append("Increase your study hours to at least 4–5 hours daily.")
    if attendance < 75:
        tips.append("Maintain above 80% attendance to improve consistency.")
    if test_score < 60:
        tips.append("Revise difficult topics and take mock tests weekly.")
    if focus_duration < 45:
        tips.append("Your focus span is short — try the Pomodoro technique (25 min focused, 5 min break).")
    if distraction > 6:
        tips.append("Avoid distractions — study in a quiet environment and keep your phone away while studying.")

    if not tips:
        st.success("🔥 Excellent! Your study habits are strong. Keep it up!")
    else:
        for t in tips:
            st.markdown(f"- {t}")

    # --- SUBJECT RECOMMENDATION ---
    st.markdown("### 📘 Subject Recommendation Based on Your Inputs")
    if not tenth_passed:
        st.info("We recommend you complete 10th first before choosing a stream.")
    else:
        # Simple recommendation based on focus, test score, and interest
        if test_score >= 80 and focus_duration >= 60:
            recommended_subject = "Science"
        elif test_score >= 60 and test_score < 80:
            recommended_subject = "Commerce"
        else:
            recommended_subject = "Arts"
        
        st.success(f"✅ Based on your inputs, you may consider: **{recommended_subject}** stream")

        # --- COURSE RECOMMENDATION ---
        st.markdown("### 🎓 Recommended Courses for Your Stream")
        if not courses_df.empty:
            df_filtered = courses_df[
                (courses_df["subject"].str.lower() == stream.lower()) &
                (courses_df["level"].str.lower() == course_label.lower())
            ]
            if not df_filtered.empty:
                for _, row in df_filtered.iterrows():
                    st.markdown(f"**{row['name']}**  \n📘 *{row['subject']} | {row['level']}*  \n🔗 [Go to Course](https://www.coursera.org/learn/{row['slug']})")
                    st.markdown("---")
            else:
                st.info("No courses found for your selected stream and recommended level.")
        else:
            st.warning("⚠️ No dataset found. Please add 'dataset/courses.csv'.")
