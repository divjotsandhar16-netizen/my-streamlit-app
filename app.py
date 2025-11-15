import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ================================================================
# CORPORATE PREMIUM CSS
# ================================================================
st.markdown("""
<style>

:root {
    --primary: #1a73e8;
    --primary-light: #e8f0fe;
    --bg: #f4f6f9;
    --card-bg: #ffffff;
    --text-dark: #1f1f1f;
    --shadow: 0 4px 18px rgba(0,0,0,0.08);
}

/* full background */
.stApp {
    background: var(--bg);
}

/* main title */
.main-title {
    font-size: 48px;
    font-weight: 900;
    text-align: center;
    color: var(--primary);
    margin-top: 10px;
}

/* card */
.card {
    background: var(--card-bg);
    padding: 22px;
    border-radius: 14px;
    box-shadow: var(--shadow);
    border: 1px solid #e5e7eb;
    margin-bottom: 18px;
}

/* buttons */
.stButton > button {
    background-color: var(--primary);
    color: white;
    font-weight: bold;
    padding: 10px 26px;
    border-radius: 8px;
    border: none;
    box-shadow: var(--shadow);
}
.stButton > button:hover {
    background-color: #0d62cf;
}

/* sidebar clean look */
.css-1d391kg, .css-12ttj6m, .css-1v3fvcr {
    background-color: #fff !important;
    padding: 10px;
    border-right: 1px solid #e5e7eb;
}

</style>
""", unsafe_allow_html=True)



# ================================================================
# TITLE
# ================================================================
st.markdown("<h1 class='main-title'>Heart Disease Prediction Dashboard</h1>", unsafe_allow_html=True)
st.write("<p style='text-align:center;'>Professional ML Dashboard for Data Analysis, Model Training & Predictions.</p>", unsafe_allow_html=True)



# ================================================================
# SIDEBAR
# ================================================================
st.sidebar.header("⚙️ Controls")

dataset_choice = st.sidebar.selectbox("Choose Dataset", ["Use heart.csv", "Upload CSV"])
train_button = st.sidebar.button("Train Model")



# ================================================================
# LOAD DATA
# ================================================================
if dataset_choice == "Use heart.csv":
    df = pd.read_csv("data/heart.csv")
else:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        st.warning("No file uploaded — using default dataset.")
        df = pd.read_csv("data/heart.csv")

st.markdown("<div class='card'><h3>📌 Dataset Preview</h3></div>", unsafe_allow_html=True)
st.dataframe(df, use_container_width=True)



# ================================================================
# SUMMARY
# ================================================================
st.markdown("<div class='card'><h3>📊 Data Summary</h3></div>", unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    st.markdown("<div class='card'><h4>🔢 Summary Statistics</h4></div>", unsafe_allow_html=True)
    st.write(df.describe())

with c2:
    st.markdown("<div class='card'><h4>🧩 Missing Values</h4></div>", unsafe_allow_html=True)
    st.table(pd.DataFrame({"Missing": df.isna().sum()}))



# ================================================================
# HEATMAP
# ================================================================
st.markdown("<div class='card'><h3>🔥 Correlation Heatmap</h3></div>", unsafe_allow_html=True)
try:
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(), cmap="Blues", ax=ax)
    st.pyplot(fig)
except:
    st.info("Heatmap unavailable.")



# ================================================================
# SCATTER MATRIX
# ================================================================
st.markdown("<div class='card'><h3>🌈 Scatter Plot Matrix</h3></div>", unsafe_allow_html=True)
try:
    fig = px.scatter_matrix(df, dimensions=df.columns[:-1], color=df.columns[-1])
    st.plotly_chart(fig, use_container_width=True)
except:
    st.info("Scatter matrix unavailable.")



# ================================================================
# TRAIN MODEL
# ================================================================
if train_button:
    st.markdown("<div class='card'><h3>🚀 Training The Model...</h3></div>", unsafe_allow_html=True)

    if "target" not in df.columns:
        st.error("Dataset must include a column named 'target'.")
    else:
        X = df.drop("target", axis=1)
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train_scaled, y_train)

        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)

        joblib.dump({
            "model": model,
            "scaler": scaler,
            "columns": list(X.columns)
        }, "model.joblib")

        st.success(f"🎉 Model Trained Successfully — Accuracy: **{acc:.3f}**")

        st.markdown("<div class='card'><h4>📄 Classification Report</h4></div>", unsafe_allow_html=True)
        st.code(classification_report(y_test, preds))

        st.markdown("<div class='card'><h4>🔢 Confusion Matrix</h4></div>", unsafe_allow_html=True)
        st.write(confusion_matrix(y_test, preds))



# ================================================================
# PREDICTION
# ================================================================
st.markdown("<div class='card'><h3>🔮 Make Predictions</h3></div>", unsafe_allow_html=True)

if os.path.exists("model.joblib"):

    try:
        bundle = joblib.load("model.joblib")
        model = bundle["model"]
        scaler = bundle["scaler"]
        cols = bundle["columns"]
        st.success("Model Loaded Successfully!")
    except:
        st.error("Model file corrupted — retrain.")
        st.stop()

    user_data = {}

    with st.form("prediction_form"):
        for col in cols:
            val = st.number_input(col, value=float(df[col].mean()))
            user_data[col] = val

        predict_btn = st.form_submit_button("Predict")

    if predict_btn:
        X_user = pd.DataFrame([user_data])
        X_scaled = scaler.transform(X_user)
        pred = model.predict(X_scaled)[0]

        label = "No Heart Disease" if pred == 0 else "⚠️ High Risk — Needs Attention"

        st.markdown(f"<div class='card'><h2>{label}</h2></div>", unsafe_allow_html=True)

else:
    st.info("Train the model first.")
