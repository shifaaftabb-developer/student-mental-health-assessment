"""
Student Mental Health Screening System
Phase 6 - System Development
Machine Learning-Based Student Mental Health Assessment System with Comparative Model Analysis

Model: Logistic Regression (top performer in Phase 5 Comparative Analysis)
        Accuracy 96.03% | Precision 73.97% | Recall 41.06% | F1 52.81% | ROC-AUC 96.13%

Run with:  streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Mental Health Screening",
    page_icon="🧠",
    layout="centered",
)

APP_DIR = Path(__file__).parent

# --------------------------------------------------------------------------
# Load model artifacts (cached so they only load once per session)
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(APP_DIR / "logreg_model.joblib")
    scaler = joblib.load(APP_DIR / "scaler.joblib")
    with open(APP_DIR / "feature_columns.json") as f:
        feature_columns = json.load(f)
    return model, scaler, feature_columns


model, scaler, FEATURE_COLUMNS = load_artifacts()

# --------------------------------------------------------------------------
# Questionnaire definitions
# (paraphrased screening items grouped by the same constructs used in
#  Phase 3 preprocessing: anxiety, insomnia, and perceived stress, plus
#  demographics. NOT the verbatim licensed instruments - a simplified
#  version built for this educational prototype.)
# --------------------------------------------------------------------------

FREQ_4 = ["Not at all", "Several days", "More than half the days", "Nearly every day"]  # 0-3
SEVERITY_5 = ["None", "Mild", "Moderate", "Severe", "Very severe"]  # 0-4
FREQ_5 = ["Never", "Almost never", "Sometimes", "Fairly often", "Very often"]  # 1-5

GAD7_ITEMS = [
    "Feeling nervous, anxious, or on edge",
    "Not being able to stop or control worrying",
    "Worrying too much about different things",
    "Trouble relaxing",
    "Being so restless that it's hard to sit still",
    "Becoming easily annoyed or irritable",
    "Feeling afraid as if something awful might happen",
]

ISI_ITEMS = [
    "Difficulty falling asleep",
    "Difficulty staying asleep",
    "Problems waking up too early",
    "How satisfied are you with your current sleep pattern?",
    "How noticeable to others is your sleep problem in terms of impairing your quality of life?",
    "How worried/distressed are you about your current sleep problem?",
    "How much does your sleep problem interfere with your daily functioning "
    "(e.g. daytime fatigue, concentration, mood)?",
]

PSS_ITEMS = [
    "Been upset because of something that happened unexpectedly",
    "Felt unable to control the important things in your life",
    "Felt nervous and stressed",
    "Felt confident about your ability to handle personal problems",
    "Felt that things were going your way",
    "Found that you could not cope with all the things you had to do",
    "Been able to control irritations in your life",
    "Felt that you were on top of things",
    "Been angered because of things outside of your control",
    "Felt difficulties were piling up so high you could not overcome them",
    "Felt that you were effectively coping with important life changes",
    "Felt confident about handling personal problems even under pressure",
    "Felt that things were going smoothly",
    "Felt overwhelmed by the pace and demands of daily life",
]

EDU_OPTIONS = ["Associate degree", "Bachelor's degree", "Master's degree", "Doctorate degree"]
SMOKE_OPTIONS = [
    "Current smoker (cumulative smoking >10 packs)",
    "Former smoker (cumulative smoking >10 packs), but not in the past year",
    "Never smokes",
    "Occasional smoker (cumulative smoking <10 packs)",
]
DRINK_OPTIONS = [
    "Current regular drinker (more than once a week)",
    "Drank in the past (more than once a week), but not in the past year",
    "Drinks occasionally (less than once a week)",
    "Never drinks",
]

# --------------------------------------------------------------------------
# Header + disclaimer
# --------------------------------------------------------------------------
st.title("🧠 Student Mental Health Screening")
st.caption("Machine Learning-Based Student Mental Health Assessment System — Phase 6 Prototype")

st.warning(
    "**Disclaimer:** This tool is built for **educational and screening purposes only** "
    "as part of a university Machine Learning course project. It is **not a medical "
    "diagnostic tool** and must not be used as a substitute for professional psychiatric "
    "or psychological evaluation. If you or someone you know is struggling, please reach "
    "out to a qualified mental health professional or a local crisis helpline."
)

with st.expander("About this system"):
    st.write(
        "This screening system uses a **Logistic Regression** model trained on "
        "demographic information plus anxiety (GAD-7 style), insomnia (ISI style), and "
        "perceived stress (PSS style) questionnaire responses. It was selected as the "
        "final deployed model because it had the strongest overall performance "
        "(Accuracy 96.03%, F1-Score 52.81%, ROC-AUC 96.13%) among the four models "
        "compared in Phase 5 of this project: Logistic Regression, Artificial Neural "
        "Network, k-Nearest Neighbors, and Naive Bayes."
    )

# --------------------------------------------------------------------------
# Form
# --------------------------------------------------------------------------
with st.form("screening_form"):

    st.subheader("1. About You")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=15, max_value=80, value=20, step=1)
        gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
    with col2:
        edu = st.selectbox("Education level", EDU_OPTIONS, index=1)

    smoke = st.selectbox("Smoking habits", SMOKE_OPTIONS, index=2)
    drink = st.selectbox("Drinking habits", DRINK_OPTIONS, index=3)

    st.divider()
    st.subheader("2. Anxiety — over the last 2 weeks, how often have you been bothered by:")
    gad7_answers = []
    for i, item in enumerate(GAD7_ITEMS, start=1):
        val = st.select_slider(f"GAD-{i}. {item}", options=FREQ_4, value=FREQ_4[0], key=f"gad{i}")
        gad7_answers.append(FREQ_4.index(val))

    st.divider()
    st.subheader("3. Sleep — please rate your current sleep problems:")
    isi_answers = []
    for i, item in enumerate(ISI_ITEMS, start=1):
        val = st.select_slider(f"ISI-{i}. {item}", options=SEVERITY_5, value=SEVERITY_5[0], key=f"isi{i}")
        isi_answers.append(SEVERITY_5.index(val))

    st.divider()
    st.subheader("4. Stress — in the last month, how often have you:")
    pss_answers = []
    for i, item in enumerate(PSS_ITEMS, start=1):
        val = st.select_slider(f"PSS-{i}. {item}", options=FREQ_5, value=FREQ_5[0], key=f"pss{i}")
        pss_answers.append(FREQ_5.index(val) + 1)  # dataset scale is 1-5

    submitted = st.form_submit_button("Get Assessment", use_container_width=True)

# --------------------------------------------------------------------------
# Build feature vector + predict
# --------------------------------------------------------------------------
if submitted:
    row = {col: 0 for col in FEATURE_COLUMNS}
    row["age"] = float(age)

    for i, v in enumerate(gad7_answers, start=1):
        row[f"gad7_q{i}"] = v
    for i, v in enumerate(isi_answers, start=1):
        row[f"isi_q{i}"] = v
    for i, v in enumerate(pss_answers, start=1):
        row[f"pss_q{i}"] = v

    if gender == "Male":
        row["gender_male"] = True

    if edu == "Bachelor's degree":
        row["edu_bachelor's degree"] = True
    elif edu == "Master's degree":
        row["edu_master's degree"] = True
    elif edu == "Doctorate degree":
        row["edu_doctorate degree"] = True
    # Associate degree -> baseline, all edu_* stay False

    if smoke == "Former smoker (cumulative smoking >10 packs), but not in the past year":
        row["smoke_former smoker (cumulative smoking >10 packs), but not in the past year"] = True
    elif smoke == "Never smokes":
        row["smoke_never smokes"] = True
    elif smoke == "Occasional smoker (cumulative smoking <10 packs)":
        row["smoke_occasional smoker (cumulative smoking <10 packs)"] = True
    # Current smoker -> baseline

    if drink == "Drank in the past (more than once a week), but not in the past year":
        row["drink_drank in the past (more than once a week), but not in the past year"] = True
    elif drink == "Drinks occasionally (less than once a week)":
        row["drink_drinks occasionally (less than once a week)"] = True
    elif drink == "Never drinks":
        row["drink_never drinks"] = True
    # Current regular drinker -> baseline

    X = pd.DataFrame([row])[FEATURE_COLUMNS]
    X_scaled = scaler.transform(X)

    proba = model.predict_proba(X_scaled)[0, 1]
    pred = model.predict(X_scaled)[0]

    st.divider()
    st.subheader("Assessment Result")

    if pred == 1:
        st.error(f"**Predicted risk category: Distressed**  \nModel confidence: {proba:.1%}")
    else:
        st.success(f"**Predicted risk category: Non-Distressed**  \nModel confidence: {1 - proba:.1%}")

    st.progress(min(max(proba, 0.0), 1.0), text=f"Estimated distress probability: {proba:.1%}")

    st.caption(
        "This result is a statistical estimate from a machine learning model trained on "
        "survey data, not a clinical diagnosis. Please see the disclaimer above."
    )
