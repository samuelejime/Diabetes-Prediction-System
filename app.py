import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="centered"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("diabetes_prediction_pipeline.pkl")

model = load_model()

# -----------------------------
# HEADER
# -----------------------------
st.title("🩺 Diabetes Prediction System")

st.write(
    """
    Welcome!

    This application predicts whether a patient is likely to have diabetes
    based on several medical measurements.

    Fill in the patient's information below and click **Predict**.
    """
)

st.divider()

# -----------------------------
# INPUTS
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0,
        max_value=250,
        value=120
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0,
        max_value=150,
        value=70
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0,
        max_value=100,
        value=20
    )

with col2:
    insulin = st.number_input(
        "Insulin",
        min_value=0,
        max_value=900,
        value=80
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0
    )

    pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        format="%.3f"
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )

st.divider()

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.button("Predict Diabetes", use_container_width=True):

    input_data = pd.DataFrame({
        "Pregnancies":[pregnancies],
        "Glucose":[glucose],
        "BloodPressure":[blood_pressure],
        "SkinThickness":[skin_thickness],
        "Insulin":[insulin],
        "BMI":[bmi],
        "DiabetesPedigreeFunction":[pedigree],
        "Age":[age]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ High likelihood of Diabetes")
        st.progress(float(probability[1]))
        st.write(f"Confidence: **{probability[1]*100:.2f}%**")

    else:
        st.success("✅ Low likelihood of Diabetes")
        st.progress(float(probability[0]))
        st.write(f"Confidence: **{probability[0]*100:.2f}%**")

    with st.expander("View Entered Information"):
        st.dataframe(input_data, use_container_width=True)

st.divider()

st.caption(
    "⚠️ This application is intended for educational purposes only "
    "and should not replace professional medical advice."
)