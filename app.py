import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and columns
model = joblib.load("logistic_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")  # Columns used during training

# Streamlit UI
st.title("Heart Disease Prediction")
st.markdown("Provide the following details to check your heart disease risk:")

# Define healthy ranges for numeric features
healthy_ranges = {
    "Age": (18, 45),
    "RestingBP": (90, 120),
    "Cholesterol": (125, 200),
    "MaxHR": (100, 170),
    "Oldpeak": (0, 1.0)
}

def risk_indicator(value, feature):
    low, high = healthy_ranges[feature]
    if value < low:
        return f"⚠️ Low ({low}-{high} is healthy)"
    elif value > high:
        return f"⚠️ High ({low}-{high} is healthy)"
    else:
        return f"✅ Normal ({low}-{high})"

# Numeric inputs
age = st.slider("Age", 18, 100, 40)
st.caption(risk_indicator(age, "Age"))

resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
st.caption(risk_indicator(resting_bp, "RestingBP"))

cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
st.caption(risk_indicator(cholesterol, "Cholesterol"))

max_hr = st.slider("Max Heart Rate", 60, 220, 150)
st.caption(risk_indicator(max_hr, "MaxHR"))

oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st.caption(risk_indicator(oldpeak, "Oldpeak"))

# Categorical inputs with risk guidance
sex = st.selectbox("Sex", ["M", "F"])

chest_pain_risk = {
    "ATA": "✅ Low Risk (Typical Angina)",
    "NAP": "⚠️ Moderate Risk (Non-Anginal Pain)",
    "TA":  "⚠️ Moderate Risk (Atypical Angina)",
    "ASY": "⚠️ High Risk (Asymptomatic)"
}
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
st.caption(chest_pain_risk[chest_pain])

resting_ecg_risk = {
    "Normal": "✅ Low Risk",
    "ST": "⚠️ Moderate Risk (ST-T Abnormality)",
    "LVH": "⚠️ Moderate Risk (Left Ventricular Hypertrophy)"
}
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
st.caption(resting_ecg_risk[resting_ecg])

exercise_angina_risk = {
    "N": "✅ No Exercise-Induced Angina",
    "Y": "⚠️ Yes, Higher Risk"
}
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
st.caption(exercise_angina_risk[exercise_angina])

fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
st.caption("⚠️ High if 1, ✅ Normal if 0")

st_slope_risk = {
    "Up": "✅ Low Risk (Upward slope)",
    "Flat": "⚠️ Moderate Risk",
    "Down": "⚠️ High Risk"
}
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
st.caption(st_slope_risk[st_slope])

# Prediction logic
if st.button("Predict"):

    # Step 1: Initialize dataframe with all expected columns
    input_df = pd.DataFrame(columns=expected_columns)
    input_df.loc[0] = 0  # fill all with 0 initially

    # Step 2: Fill numeric features (exclude Oldpeak from scaling)
    numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR']
    input_df.loc[0, 'Age'] = age
    input_df.loc[0, 'RestingBP'] = resting_bp
    input_df.loc[0, 'Cholesterol'] = cholesterol
    input_df.loc[0, 'MaxHR'] = max_hr
    input_df.loc[0, 'Oldpeak'] = oldpeak  # keep as-is

    # Step 3: Fill categorical features
    cat_mapping = {
        'Sex': sex,
        'ChestPainType': chest_pain,
        'RestingECG': resting_ecg,
        'ExerciseAngina': exercise_angina,
        'FastingBS': fasting_bs,
        'ST_Slope': st_slope
    }

    for feature, val in cat_mapping.items():
        col_name = f"{feature}_{val}"
        if col_name in input_df.columns:
            input_df.loc[0, col_name] = 1

    # Step 4: Scale numeric features (excluding Oldpeak)
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Step 5: Predict
    prediction = model.predict(input_df)[0]

    # Step 6: Show result
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
