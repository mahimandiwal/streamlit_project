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
# NOTE: the model was trained on exactly these 5 columns, in this exact order:
# ['study_hours_per_week', 'attendance_percentage', 'final_grade',
#  'extracurricular_Yes', 'part_time_job_Yes']
# study_hours_per_week and attendance_percentage were normalized to a 0-1 scale
# in the training data (train_model.py never re-derives that scaling), so we
# collect them here as 0-100 sliders and divide by 100 to match that range.
try:
    model = joblib.load(MODEL_DIR / "model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    course_enc = joblib.load(MODEL_DIR / "course_encoder.pkl")
except Exception as e:
    st.error("❌ Some model files are missing in 'training/model'. Please make sure model.pkl, scaler.pkl, and course_encoder.pkl exist.")
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
# These 5 inputs map directly to the features the model was trained on.
with st.form("habit_form"):
    study_intensity = st.slider(
        "📚 Study Intensity (relative to peers, 0 = lowest, 100 = highest)",
        0, 100, 50
    )
    st.caption(
        f"Selected Study Intensity: {study_intensity}/100 "
        "(the training data used a relative 0-1 study-hours score rather than raw hours, "
        "so this slider mirrors that same relative scale)"
    )

    attendance = st.slider("🏫 Attendance (%)", 0, 100, 85)
    st.caption(f"Selected Attendance: {attendance}%")

    test_score = st.slider("🧾 Average Test Score / Final Grade (%)", 0, 100, 70)
    st.caption(f"Selected Test Score: {test_score}%")

    extracurricular = st.checkbox("🎭 Participates in Extracurricular Activities?")
    part_time_job = st.checkbox("💼 Has a Part-Time Job?")

    # 10th Passed checkbox (used only for the non-ML course-recommendation section below)
    tenth_passed = st.checkbox("✅ 10th Passed?")

    stream = None
    if tenth_passed:
        stream = st.selectbox("📘 Select Your Preferred Stream", ["Science", "Commerce", "Arts"])

    submitted = st.form_submit_button("🔍 Analyze My Habits")

# -------------------- ON SUBMIT -------------------- #
if submitted:
    # --- MODEL PREDICTION ---
    # Order must exactly match training: study_hours_per_week, attendance_percentage,
    # final_grade, extracurricular_Yes, part_time_job_Yes
    study_hours_per_week_norm = study_intensity / 100
    attendance_percentage_norm = attendance / 100
    final_grade = test_score  # already 0-100, matches training scale
    extracurricular_flag = 1 if extracurricular else 0
    part_time_flag = 1 if part_time_job else 0

    X = np.array([[
        study_hours_per_week_norm,
        attendance_percentage_norm,
        final_grade,
        extracurricular_flag,
        part_time_flag
    ]])

    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    course_label = course_enc.inverse_transform([pred])[0]

    st.success(f"🎯 Recommended Course Level: **{course_label}**")

    # --- PERFORMANCE SCORECARD ---
    # Rebuilt using only the real inputs (no more fake focus/distraction numbers)
    st.markdown("### 📊 Your Performance Scorecard")
    study_score = round(study_intensity / 10, 1)
    consistency_score = round(attendance / 10, 1)
    test_prep_score = round(test_score / 10, 1)

    col1, col2, col3 = st.columns(3)
    col1.metric("📚 Study Intensity", f"{study_score}/10")
    col2.metric("📅 Consistency", f"{consistency_score}/10")
    col3.metric("🧠 Test Preparation", f"{test_prep_score}/10")

    overall = round((study_score + consistency_score + test_prep_score) / 3, 1)
    if overall >= 8:
        st.success(f"🌟 Overall Score: {overall}/10 — Excellent Performance!")
    elif overall >= 5:
        st.info(f"👍 Overall Score: {overall}/10 — Good, but can improve.")
    else:
        st.warning(f"⚠️ Overall Score: {overall}/10 — Needs improvement in study habits.")

    # --- STUDY IMPROVEMENT ADVICE ---
    st.markdown("### 🧠 Personalized Study Advice")
    tips = []
    if study_intensity < 40:
        tips.append("Increase your study time relative to your peers — aim for consistent daily study blocks.")
    if attendance < 75:
        tips.append("Maintain above 80% attendance to improve consistency.")
    if test_score < 60:
        tips.append("Revise difficult topics and take mock tests weekly.")
    if part_time_job and study_intensity < 50:
        tips.append("Balancing a part-time job with studies can be tough — try scheduling fixed study blocks around work hours.")
    if not extracurricular and overall < 5:
        tips.append("Consider light extracurricular involvement — it's linked with better overall engagement in the training data.")

    if not tips:
        st.success("🔥 Excellent! Your study habits are strong. Keep it up!")
    else:
        for t in tips:
            st.markdown(f"- {t}")

    # --- SUBJECT RECOMMENDATION (rule-based, independent of the ML model) --- #
    st.markdown("### 📘 Subject Recommendation Based on Your Inputs")
    if not tenth_passed:
        st.info("We recommend you complete 10th first before choosing a stream.")
    else:
        if test_score >= 80 and study_intensity >= 60:
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
