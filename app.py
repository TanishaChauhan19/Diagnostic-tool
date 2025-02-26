import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Load trained models
heart_model = joblib.load(r"heart_model_path")
diabetes_model = joblib.load(r"diabetes_model_path")
diabetes_scaler = joblib.load(r"diabetes_scalar_model_path")
pneumonia_model = tf.keras.models.load_model(r"pneomonia_model_path")  # Load pneumonia model

# Pneumonia class labels
pneumonia_classes = ["Normal", "Pneumonia"]

# Streamlit UI
st.title("Multi-Disease Diagnostic Tool 🏥")

# Select disease type
disease = st.selectbox("Select Disease for Prediction", ["Heart Disease", "Diabetes", "Hepatitis", "Pneumonia"])

# Heart Disease Prediction
if disease == "Heart Disease":
    st.subheader("Heart Disease Prediction")
    age = st.number_input("Age", min_value=1, max_value=120, value=50, key="heart_age")
    sex = st.radio("Sex", ["Male", "Female"], key="heart_sex")
    chest_pain_type = st.selectbox("Chest Pain Type", ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"], key="heart_cp")
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=200, value=120, key="heart_bp")
    cholestoral = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=400, value=200, key="heart_chol")
    fasting_blood_sugar = st.radio("Fasting Blood Sugar", ["Lower than 120 mg/ml", "Greater than 120 mg/ml"], key="heart_fbs")
    rest_ecg = st.selectbox("Rest ECG", ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"], key="heart_ecg")
    max_heart_rate = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=150, key="heart_hr")
    exercise_angina = st.radio("Exercise Induced Angina", ["Yes", "No"], key="heart_angina")
    oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=6.0, value=1.0, key="heart_oldpeak")
    slope = st.selectbox("Slope", ["Upsloping", "Flat", "Downsloping"], key="heart_slope")
    vessels_colored = st.selectbox("Vessels Colored by Fluoroscopy", ["Zero", "One", "Two", "Three"], key="heart_vessels")
    thalassemia = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversable Defect"], key="heart_thal")

    # Encoding categorical variables
    sex = 1 if sex == "Male" else 0
    fasting_blood_sugar = 1 if fasting_blood_sugar == "Greater than 120 mg/ml" else 0
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    chest_pain_type = {"Typical angina": 0, "Atypical angina": 1, "Non-anginal pain": 2, "Asymptomatic": 3}[chest_pain_type]
    rest_ecg = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}[rest_ecg]
    slope = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}[slope]
    vessels_colored = {"Zero": 0, "One": 1, "Two": 2, "Three": 3}[vessels_colored]
    thalassemia = {"Normal": 0, "Fixed Defect": 1, "Reversable Defect": 2}[thalassemia]

    input_data = np.array([[age, sex, chest_pain_type, resting_bp, cholestoral,
                             fasting_blood_sugar, rest_ecg, max_heart_rate, exercise_angina,
                             oldpeak, slope, vessels_colored, thalassemia]])
    
    if st.button("Predict Heart Disease Risk"):
        prediction = heart_model.predict(input_data)[0]
        st.success(f"Predicted Risk: {'High' if prediction == 1 else 'Low'}")

# Diabetes Prediction
elif disease == "Diabetes":
    st.subheader("Diabetes Prediction")
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=6, key="diabetes_preg")
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=148, key="diabetes_glucose")
    bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=72, key="diabetes_bp")
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=35, key="diabetes_skin")
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=0, key="diabetes_insulin")
    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=33.6, key="diabetes_bmi")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.627, key="diabetes_dpf")
    age = st.number_input("Age", min_value=1, max_value=120, value=50, key="diabetes_age")

    input_data = np.array([pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]).reshape(1, -1)
    std_data = diabetes_scaler.transform(input_data)
    
    if st.button("Predict Diabetes"):
        prediction = diabetes_model.predict(std_data)[0]
        st.success("Non-Diabetic 🟢" if prediction == 0 else "Diabetic 🔴")

# Pneumonia Prediction
elif disease == "Pneumonia":
    st.subheader("Pneumonia Prediction using X-ray Image")
    uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if st.button("Predict Pneumonia"):
            prediction = pneumonia_model.predict(img_array)[0][0]
            predicted_class = pneumonia_classes[int(prediction > 0.6)]
            confidence = prediction * 100 if prediction > 0.6 else (1 - prediction) * 100
            st.success(f"Prediction: {predicted_class} (Confidence: {confidence:.2f}%)")

