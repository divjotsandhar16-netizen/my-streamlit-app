# app.py - Professional Heart Disease Dashboard (Neon+Dark+Professional)
# Replace your existing app.py with this file. It is designed to be polished, compact,
# and feature-rich while still easy to run locally.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve

# ------------------------- Page config -------------------------
st.set_page_config(page_title="Heart Disease — Professional Dashboard",
                   layout="wide",
                   page_icon="💓")

# ------------------------- Styles (professional glass + accent) -------------------------
st.markdown("""
<style>
/* page background */
[data-testid="stAppViewContainer"] {
  background: linear-gradient(180deg,#eef2ff 0%, #dbe4ff 60%);
}
.block-container{ padding:1rem 2rem !important }

/* header */
.header-title{
  font-size:34px !important;
  font-weight:800 !important;
  color:#E6F0FF;
  text-shadow: 0 4px 18px rgba(74, 0, 224, 0.18);
}
.header-sub{
  color: #bfc9ff;
}

/* card */
.card { background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015));
        border-radius:16px; padding:18px; border:1px solid rgba(255,255,255,0.04);}

/* sidebar */
[data-testid="stSidebar"] {background: linear-gradient(180deg,#e6e9ff,#cfd8ff); border-right:1px solid rgba(0,0,0,0.1);}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {color:#e6f0ff}

/* buttons */
.stButton>button{ background: linear-gradient(90deg,#6d28d9,#4a00e0); border-radius:10px; color:white; padding:8px 14px; border:none}
.stButton>button:hover{ transform:scale(1.02)}

</style>
""", unsafe_allow_html=True)

# ------------------------- Helper utilities -------------------------
DEFAULT_DATA_PATH = "data/heart.csv"
# Also accept uploaded system path (uploaded by user); developer: provide path when embedding in system
FALLBACK_UPLOAD_PATH = "/mnt/data/heart.csv.csv"

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

def save_model_bytes(bundle):
    buffer = BytesIO()
    joblib.dump(bundle, buffer)
    buffer.seek(0)
    return buffer

# ------------------------- Layout -------------------------
with st.container():
    col1, col2 = st.columns([8,2])
    with col1:
        st.markdown("<div class='header-title'>💙 Heart Disease Prediction — Professional Dashboard</div>", unsafe_allow_html=True)
        st.markdown("<div class='header-sub'>A polished, explainable dashboard for dataset exploration, model training and live predictions.</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div style='text-align:right;color:#d6dffb'>v1.0 • Advanced UI</div>", unsafe_allow_html=True)

st.sidebar.title("REFER HERE:")
page = st.sidebar.radio("Go to:", ["Dataset", "EDA & Visuals", "Train & Evaluate", "Predict & Export"], index=0)
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit — Professional layout & advanced metrics")

# ------------------------- DATASET PAGE -------------------------
if page == "Dataset":
    st.subheader("Upload or use built-in dataset")

    colA, colB = st.columns([3,1])
    with colA:
        uploaded = st.file_uploader("Upload CSV (or use default)", type=["csv"], help="CSV with a 'target' column is expected")
        use_default = st.checkbox("Use project dataset (data/heart.csv)", value=True)

        if uploaded:
            df = pd.read_csv(uploaded)
            st.success("File uploaded and loaded into session")
            st.session_state['data'] = df
        else:
            # try fallback path first (if user uploaded through chat)
            if use_default:
                if os.path.exists(DEFAULT_DATA_PATH):
                    df = load_csv(DEFAULT_DATA_PATH)
                    st.session_state['data'] = df
                    st.info(f"Loaded default file: {DEFAULT_DATA_PATH}")
                elif os.path.exists(FALLBACK_UPLOAD_PATH):
                    df = load_csv(FALLBACK_UPLOAD_PATH)
                    st.session_state['data'] = df
                    st.info(f"Loaded fallback file: {FALLBACK_UPLOAD_PATH}")
                else:
                    st.warning("No default file found. Please upload a CSV.")
                    df = None
            else:
                st.info("Please upload a file or select a default dataset")
                df = None

    with colB:
        if 'data' in st.session_state:
            st.download_button("Download CSV", data=pd.read_csv(DEFAULT_DATA_PATH).to_csv(index=False), file_name="heart.csv", mime="text/csv")

    if 'data' in st.session_state:
        st.markdown("### Dataset preview")
        st.dataframe(st.session_state['data'].head(10))
        st.markdown("#### Columns & types")
        dtypes = pd.DataFrame(st.session_state['data'].dtypes, columns=['dtype'])
        dtypes['non_null'] = st.session_state['data'].notnull().sum().values
        st.table(dtypes)

# ------------------------- EDA PAGE -------------------------
elif page == "EDA & Visuals":
    if 'data' not in st.session_state:
        st.warning("Upload dataset first on the Dataset page.")
    else:
        df = st.session_state['data'].copy()
        st.subheader("Exploratory Data Analysis — interactive charts")

        # top KPIs
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Rows", f"{df.shape[0]:,}")
        with c2:
            st.metric("Columns", f"{df.shape[1]}")
        with c3:
            if 'target' in df.columns:
                pos = int(df['target'].sum())
                st.metric("Positive (target=1)", f"{pos}")
            else:
                st.metric("Positive (target=1)", "N/A")
        with c4:
            st.metric("Missing cells", f"{df.isna().sum().sum()}")

        st.markdown("---")

        # Distribution & class balance
        st.write("### Class balance & distributions")
        if 'target' in df.columns:
            fig = px.histogram(df, x='target', title='Class distribution', text_auto=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No 'target' column found — class plots unavailable")

        # Correlation heatmap (plotly)
        st.write("### Correlation heatmap (interactive)")
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] >= 2:
            corr = num_df.corr()
            heat = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='Blues'))
            heat.update_layout(height=500, margin=dict(l=40,r=40,t=40,b=40))
            st.plotly_chart(heat, use_container_width=True)

        # Pairwise scatter selector
        st.write("### Pairwise scatter — choose two features")
        cols = list(num_df.columns)
        if len(cols) >= 2:
            left, right = st.columns(2)
            with left:
                x = st.selectbox("X", cols, index=0)
            with right:
                y = st.selectbox("Y", cols, index=1)
            color = st.selectbox("Color by (optional)", ['None'] + cols, index=0)
            if color == 'None':
                fig = px.scatter(num_df, x=x, y=y, title=f"{x} vs {y}")
            else:
                fig = px.scatter(num_df, x=x, y=y, color=color, title=f"{x} vs {y} colored by {color}")
            st.plotly_chart(fig, use_container_width=True)

        # Feature boxplots
        st.write("### Boxplots (quick overview of ranges)")
        sel = st.multiselect("Choose numeric columns", cols, default=cols[:4])
        if sel:
            for col in sel:
                fig = px.box(num_df, y=col, points='outliers', title=f"{col} — Boxplot")
                st.plotly_chart(fig, use_container_width=True)

# ------------------------- TRAIN & EVAL PAGE -------------------------
elif page == "Train & Evaluate":
    st.subheader("Train model and evaluate metrics")
    if 'data' not in st.session_state:
        st.warning("Upload dataset first on the Dataset page.")
    else:
        df = st.session_state['data'].copy()

        if 'target' not in df.columns:
            st.error("Dataset must contain a 'target' column to train. Use Dataset page to upload a proper CSV.")
        else:
            X = df.drop(columns=['target'])
            y = df['target']

            seed = st.number_input("Random seed", value=42)
            test_size = st.slider("Test size", 5, 50, 20)
            n_estimators = st.slider("RandomForest - n_estimators", 10, 500, 150)

            if st.button("Train model"):
                # train/test without stratify if class issue; try stratify if feasible
                try:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=seed, stratify=y)
                except Exception:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=seed)

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
                model.fit(X_train_scaled, y_train)

                preds = model.predict(X_test_scaled)
                acc = accuracy_score(y_test, preds)

                st.success(f"Model trained — Accuracy: {acc:.3f}")

                # show classification report
                st.subheader("Classification report")
                st.text(classification_report(y_test, preds, zero_division=0))

                # confusion matrix
                cm = confusion_matrix(y_test, preds)
                cm_fig = px.imshow(cm, text_auto=True, labels=dict(x='Predicted', y='Actual'), title='Confusion matrix')
                st.plotly_chart(cm_fig, use_container_width=True)

                # ROC
                if len(np.unique(y_test)) == 2:
                    probs = model.predict_proba(X_test_scaled)[:,1]
                    fpr, tpr, _ = roc_curve(y_test, probs)
                    roc_auc = auc(fpr, tpr)
                    roc_fig = go.Figure()
                    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC AUC={roc_auc:.3f}'))
                    roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), showlegend=False))
                    roc_fig.update_layout(title='ROC Curve', xaxis_title='FPR', yaxis_title='TPR')
                    st.plotly_chart(roc_fig, use_container_width=True)

                # feature importance
                st.subheader('Feature importance (mean decrease impurity)')
                fi = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
                fig = px.bar(fi, x='importance', y='feature', orientation='h', title='Feature importances')
                st.plotly_chart(fig, use_container_width=True)

                # save bundle
                bundle = { 'model': model, 'scaler': scaler, 'columns': list(X.columns) }
                st.session_state['bundle'] = bundle
                st.success('Model saved to session — you can export the bundle')
                st.download_button('Export model (.joblib)', data=save_model_bytes(bundle), file_name='model.joblib')

# ------------------------- PREDICT & EXPORT PAGE -------------------------
elif page == "Predict & Export":
    st.subheader("Interactive prediction form")

    if 'bundle' not in st.session_state:
        if os.path.exists('model.joblib'):
            try:
                bundle = joblib.load('model.joblib')
                st.session_state['bundle'] = bundle
            except Exception:
                st.info('No trained model in session. Train or load model first.')

    if 'bundle' not in st.session_state:
        st.info('Train a model (Train & Evaluate) or upload an exported model (.joblib) to make predictions')
        uploaded_model = st.file_uploader('Upload .joblib model', type=['joblib'])
        if uploaded_model:
            bundle = joblib.load(uploaded_model)
            st.session_state['bundle'] = bundle
            st.success('Model loaded')
    else:
        bundle = st.session_state['bundle']
        model = bundle['model']
        scaler = bundle['scaler']
        cols = bundle['columns']

        st.write('### Fill patient features (use realistic values)')
        user = {}
        cols_left, cols_right = st.columns([1,1])
        for i, c in enumerate(cols):
            if i % 2 == 0:
                user[c] = cols_left.number_input(c, value=float(0 if c not in st.session_state.get('data', pd.DataFrame()).columns else st.session_state['data'][c].mean()))
            else:
                user[c] = cols_right.number_input(c, value=float(0 if c not in st.session_state.get('data', pd.DataFrame()).columns else st.session_state['data'][c].mean()))

        if st.button('Predict'):
            X_user = pd.DataFrame([user])[cols]
            X_scaled = scaler.transform(X_user)
            pred = model.predict(X_scaled)[0]
            prob = model.predict_proba(X_scaled)[0]
            conf = float(prob.max())
            if pred == 1:
                st.error(f'Prediction: HIGH RISK (confidence {conf:.2f})')
            else:
                st.success(f'Prediction: LOW RISK (confidence {conf:.2f})')

            # show breakdown (probabilities)
            st.write('Probabilities:')
            prob_df = pd.DataFrame({'class': list(range(len(prob))), 'prob': prob})
            st.table(prob_df)

            # allow export of input + prediction
            out = X_user.copy()
            out['prediction'] = int(pred)
            out['confidence'] = conf
            csv = out.to_csv(index=False)
            st.download_button('Download prediction as CSV', data=csv, file_name='prediction.csv')

# ------------------------- End -------------------------


# Footer
st.markdown("---")
st.markdown('<div style="text-align:center;color:#9aa8ff">Prepared by Divjot Kaur Sandhar — Clean, explainable and professional ML dashboard</div>', unsafe_allow_html=True)
