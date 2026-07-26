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
# The model was retrained to fix two issues with the original version:
# 1. It no longer uses final_grade as an input feature (final_grade was used to
#    build the label itself, so including it caused label leakage).
# 2. "Advanced" was merged into "Intermediate" -- the original data had only 9/500
#    rows labeled Advanced, too few to reliably learn as a separate class.
# feature_order.pkl stores the exact column order the model expects.
try:
    model = joblib.load(MODEL_DIR / "model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    course_enc = joblib.load(MODEL_DIR / "course_encoder.pkl")
    feature_order = joblib.load(MODEL_DIR / "feature_order.pkl")
except Exception as e:
    st.error("❌ Some model files are missing in 'training/model'. Please make sure model.pkl, scaler.pkl, course_encoder.pkl, and feature_order.pkl exist.")
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
# These 7 inputs map directly to feature_order, the exact columns the model was trained on:
# ['study_hours_per_week', 'attendance_percentage', 'sleep_hours_per_day',
#  'assignments_completed', 'extracurricular_Yes', 'part_time_job_Yes', 'internet_access_Yes']
with st.form("habit_form"):
    st.markdown("#### 📈 Inputs used by the ML model")

    study_intensity = st.slider(
        "📚 Study Intensity (relative to peers, 0 = lowest, 100 = highest)", 0, 100, 50
    )
    st.caption(
        "The training data used a relative 0-1 study-hours score rather than raw hours, "
        "so this slider mirrors that same relative scale."
    )

    attendance = st.slider("🏫 Attendance (%)", 0, 100, 85)

    sleep_quality = st.slider(
        "😴 Sleep Consistency (relative to peers, 0 = lowest, 100 = highest)", 0, 100, 50
    )
    st.caption("Same relative-scale note applies here as with Study Intensity.")

    assignments_completed_pct = st.slider("📝 Assignments Completed (%)", 0, 100, 75)

    extracurricular = st.checkbox("🎭 Participates in Extracurricular Activities?")
    part_time_job = st.checkbox("💼 Has a Part-Time Job?")
    internet_access = st.checkbox("🌐 Has Reliable Internet Access?", value=True)

    st.markdown("---")
    st.markdown("#### 📘 Inputs used only for the stream/subject recommendation below (not the ML model)")
    test_score = st.slider("🧾 Average Test Score (%)", 0, 100, 70)
    tenth_passed = st.checkbox("✅ 10th Passed?")

    stream = None
    if tenth_passed:
        stream = st.selectbox("Select Your Preferred Stream", ["Science", "Commerce", "Arts"])

    submitted = st.form_submit_button("🔍 Analyze My Habits")

# -------------------- ON SUBMIT -------------------- #
if submitted:
    # --- MODEL PREDICTION ---
    feature_values = {
        "study_hours_per_week": study_intensity / 100,
        "attendance_percentage": attendance / 100,
        "sleep_hours_per_day": sleep_quality / 100,
        "assignments_completed": assignments_completed_pct / 100,
        "extracurricular_Yes": 1 if extracurricular else 0,
        "part_time_job_Yes": 1 if part_time_job else 0,
        "internet_access_Yes": 1 if internet_access else 0,
    }
    # Build the array using feature_order so it always matches training, regardless
    # of the order fields are defined above.
    X = np.array([[feature_values[col] for col in feature_order]])

    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    course_label = course_enc.inverse_transform([pred])[0]
    pred_proba = model.predict_proba(X_scaled)[0]
    confidence = dict(zip(course_enc.classes_, pred_proba.round(2)))

    st.success(f"🎯 Recommended Course Level: **{course_label}**")
    st.caption(f"Model confidence: {confidence}")

    # --- PERFORMANCE SCORECARD (simple descriptive stats, not the ML model) --- #
    st.markdown("### 📊 Your Performance Scorecard")
    study_score = round(study_intensity / 10, 1)
    consistency_score = round(attendance / 10, 1)
    sleep_score = round(sleep_quality / 10, 1)

    col1, col2, col3 = st.columns(3)
    col1.metric("📚 Study Intensity", f"{study_score}/10")
    col2.metric("📅 Attendance", f"{consistency_score}/10")
    col3.metric("😴 Sleep Consistency", f"{sleep_score}/10")

    overall = round((study_score + consistency_score + sleep_score) / 3, 1)
    if overall >= 8:
        st.success(f"🌟 Overall Score: {overall}/10 — Excellent Performance!")
    elif overall >= 5:
        st.info(f"👍 Overall Score: {overall}/10 — Good, but can improve.")
    else:
        st.warning(f"⚠️ Overall Score: {overall}/10 — Needs improvement in study habits.")

    # --- STUDY IMPROVEMENT ADVICE (rule-based, independent of the ML model) --- #
    st.markdown("### 🧠 Personalized Study Advice")
    tips = []
    if study_intensity < 40:
        tips.append("Increase your study time relative to your peers — aim for consistent daily study blocks.")
    if attendance < 75:
        tips.append("Maintain above 80% attendance to improve consistency.")
    if sleep_quality < 40:
        tips.append("Prioritize consistent sleep — it's linked with better focus and retention.")
    if assignments_completed_pct < 60:
        tips.append("Try to complete more assignments on time — they reinforce what you study.")
    if part_time_job and study_intensity < 50:
        tips.append("Balancing a part-time job with studies can be tough — try scheduling fixed study blocks around work hours.")
    if not internet_access:
        tips.append("Limited internet access can restrict resources — check if your institution offers offline materials.")

    if not tips:
        st.success("🔥 Excellent! Your study habits are strong. Keep it up!")
    else:
        for t in tips:
            st.markdown(f"- {t}")

    # --- SUBJECT RECOMMENDATION (rule-based, uses test_score, independent of the ML model) --- #
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
                (courses_df["subject"].str.lower() == recommended_subject.lower()) &
                (courses_df["level"].str.lower() == course_label.lower())
            ]
            if not df_filtered.empty:
                for _, row in df_filtered.iterrows():
                    search_query = row['name'].replace(' ', '%20')
                    st.markdown(f"**{row['name']}**  \n📘 *{row['subject']} | {row['level']}*  \n🔗 [Search on Coursera](https://www.coursera.org/search?query={search_query})")
                    st.markdown("---")
            else:
                st.info("No courses found for your recommended subject and level.")
        else:
            st.warning("⚠️ No dataset found. Please add 'dataset/courses.csv'.")
