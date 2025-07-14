import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image
import time
import matplotlib.pyplot as plt

# --------------------- PAGE CONFIG ---------------------
st.set_page_config(
    page_title="CyberGuard - Threat Detection App",
    page_icon="logo_1.png", 
    layout="wide"
)

# --------------------- LOAD MODELS ---------------------
@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load("logistic_regression_model.joblib"),
        "Random Forest": joblib.load("random_forest_model.joblib"),
        "XGBoost": joblib.load("xgboost_model.joblib"),
        "KNN": joblib.load("knn_model.joblib"),
        "Neural Network": tf.keras.models.load_model("neural_network_model.h5")
    }

@st.cache_resource
def load_lstm():
    return tf.keras.models.load_model("lstm_cyber_threat_model.h5")

@st.cache_resource
def load_preprocessor():
    return joblib.load("preprocessor.pkl")

models = load_models()
lstm_model = load_lstm()
preprocessor = load_preprocessor()

# --------------------- HEADER ---------------------
col_logo, col_title = st.columns([1, 7])
with col_logo:
    st.image("logo_1.png", width=160)
with col_title:
    st.markdown("<h1 style='color: #2DCBFF; margin-top: 10px;'>Detect. Defend. Dominate Cyber Threats Effortlessly.</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='font-size:16px;'>Because every second counts in cybersecurity — trust our app to keep you secure.</h2>", unsafe_allow_html=True)

st.markdown("---")

# --------------------- SIDEBAR ---------------------
with st.sidebar:
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("logo_1.png", width=60)
    with col2:
        st.markdown("""
        <div style='display:flex; align-items:center; height:60px;'>
            <h3 style='color: #2DCBFF; font-size: 24px; margin: 0;'>CyberGuard</h3>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<p style='color: #2DCBFF; font-size:18px; font-weight:bold; margin-top: 4px;'>Digital Guard Against Evolving Cyber Attacks</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 📝 Read Instructions Carefully")
    st.markdown("""
    - **Upload CSV file** or use manual input  
    - **Select a model** from the dropdown  
    - **Click 'Predict Threat'** to analyze the input  
    - Output will show **Threat Probability**  
    - Result will state: **Threat Detected / No Threat**  
    """)

    st.markdown("🎓 Built for Final Year MSc Project")
    st.markdown("[GitHub Repo](https://github.com) | [Docs](#)", unsafe_allow_html=True)

# --------------------- INPUT SECTION ---------------------
st.header("🛡️ Follow the steps below and let CyberGuard secure your system.")

selected_model = st.selectbox("🛆 Choose a Model", list(models.keys()))

uploaded_file = st.file_uploader("Upload input CSV (one row only):", type=['csv'])

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully.")
else:
    with st.form("manual_input"):
        col1, col2, col3 = st.columns(3)

        with col1:
            bytes_sent = st.number_input("Bytes Sent", value=920)
            failed_logins = st.number_input("Failed Logins", value=5)
            protocol = st.selectbox("Protocol", ['HTTP', 'HTTPS', 'SSH', 'RDP', 'DNS'])
            session_duration = st.number_input("Session Duration (sec)", value=300)

        with col2:
            cpu_usage = st.slider("CPU Usage (%)", 0, 100, 78)
            user = st.text_input("User", value="user24")
            auth_method = st.selectbox("Auth Method", ['password', 'key', 'certificate'])
            memory_usage = st.slider("Memory Usage (%)", 0, 100, 45)

        with col3:
            device_id = st.text_input("Device ID", value="workstation-12")
            hour = st.slider("Hour", 0, 23, 2)
            login_success_rate = st.number_input("Login Success Rate", value=0.4)
            disk_io = st.number_input("Disk I/O", value=50.0)
            request_frequency = st.number_input("Request Frequency", value=5.0)

        is_working_hours = 1 if 8 <= hour <= 17 else 0
        source_ip = "192.168.1.1"
        destination_ip = "10.0.0.1"
        bytes_per_packet = bytes_sent / 30

        submit = st.form_submit_button("💡 Predict Threat")

    if submit:
        input_dict = {
            'bytes_sent': bytes_sent,
            'failed_logins': failed_logins,
            'cpu_usage': cpu_usage,
            'protocol': protocol,
            'user': user,
            'auth_method': auth_method,
            'device_id': device_id,
            'hour': hour,
            'is_working_hours': is_working_hours,
            'source_ip': source_ip,
            'destination_ip': destination_ip,
            'bytes_per_packet': bytes_per_packet,
            'login_success_rate': login_success_rate,
            'session_duration': session_duration,
            'memory_usage': memory_usage,
            'disk_io': disk_io,
            'request_frequency': request_frequency
        }
        input_df = pd.DataFrame([input_dict])
    else:
        input_df = None

# --------------------- ETHICS / PRIVACY NOTICE ---------------------
st.markdown("---")
with st.expander("⚠️ Important: Security, Ethical & Privacy Notice"):
    st.markdown("""
    - 📌 **This application is strictly for academic and demonstration use.**  
    - 🔒 **No personal data is stored**, transmitted, or logged.  
    - 🚫 Do not upload any real, confidential, or sensitive data.  
    - 📣 This project follows responsible AI principles, including fairness and transparency.    
    """)

# --------------------- PREDICTION ---------------------
if input_df is not None:
    with st.spinner("🔍 Analyzing for threat..."):
        time.sleep(1.5)
        try:
            X = preprocessor.transform(input_df)

            if selected_model == "Neural Network":
                X = X.toarray() if hasattr(X, "toarray") else X
                prediction = models[selected_model].predict(X)[0][0]
            elif selected_model in ["XGBoost", "KNN"]:
                X = X.toarray() if hasattr(X, "toarray") else X
                prediction = models[selected_model].predict_proba(X)[0][1]
            else:
                prediction = models[selected_model].predict_proba(X)[0][1]
        except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

            # ----- RESULT DISPLAY -----
         

import matplotlib.pyplot as plt
import numpy as np

if input_df is not None:
    with st.spinner("🔍 Analyzing for threat..."):
        time.sleep(1.5)
        try:
            X = preprocessor.transform(input_df)

            if selected_model == "Neural Network":
                X = X.toarray() if hasattr(X, "toarray") else X
                prediction = models[selected_model].predict(X)[0][0]
            elif selected_model in ["XGBoost", "KNN"]:
                X = X.toarray() if hasattr(X, "toarray") else X
                prediction = models[selected_model].predict_proba(X)[0][1]
            else:
                prediction = models[selected_model].predict_proba(X)[0][1]

            st.markdown("---")
            st.subheader(f"📊 Prediction Result using {selected_model}")
            st.metric("Threat Probability", f"{prediction:.4f}")

            # -------- Four Visualization Subplots --------
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))

            # 1. Doughnut Chart (Pie with hole)
            axs[0, 0].pie([prediction, 1 - prediction],
                          startangle=90,
                          colors=["#FF4B4B" if prediction >= 0.5 else "#4CAF50", "#E0E0E0"],
                          radius=1,
                          wedgeprops=dict(width=0.4, edgecolor='white'))
            axs[0, 0].text(0, 0, f"{int(prediction * 100)}%", ha='center', va='center', fontsize=14, color="#333")
            axs[0, 0].set_title("Threat Probability (Donut Chart)")
            axs[0, 0].axis('equal')

            # 2. Bar Chart
            axs[0, 1].bar(["Threat", "Safe"], [prediction, 1 - prediction],
                          color=["#FF4B4B", "#4CAF50"])
            axs[0, 1].set_ylim(0, 1)
            axs[0, 1].set_ylabel("Probability")
            axs[0, 1].set_title("Threat vs Safe (Bar Chart)")

            # 3. Horizontal Gauge-like bar
            axs[1, 0].barh(["Threat Level"], [prediction], color="#FF9800" if prediction >= 0.5 else "#4CAF50")
            axs[1, 0].set_xlim(0, 1)
            axs[1, 0].set_title("Threat Gauge")
            axs[1, 0].set_xlabel("Probability")

            # 4. Simulated Line Chart (current prediction trend)
            trend = [prediction * np.random.uniform(0.95, 1.05) for _ in range(10)]
            axs[1, 1].plot(trend, marker='o', color="#2196F3")
            axs[1, 1].set_ylim(0, 1)
            axs[1, 1].set_title("Simulated Threat Trend")
            axs[1, 1].set_xlabel("Time")
            axs[1, 1].set_ylabel("Threat Probability")

            plt.tight_layout()
            st.pyplot(fig)

            # Threat level message
            if prediction >= 0.8:
                st.error("🔴 High Threat Detected – Immediate Action Required!")
            elif prediction >= 0.5:
                st.warning("🟠 Moderate Threat Detected – Monitor Closely.")
            else:
                st.success("🟢 No Threat Detected – System Appears Safe.")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")



# --------------------- FOOTER ---------------------
st.markdown("---")
footer = """
<div style="text-align:center; color:gray; font-size:14px;">
    &copy; 2025 Cyber Threat App | Built for Final Year Project <br>
    <a href="mailto:areebaiftikhar921@example.com">Contact</a> | <a href="https://github.com" target="_blank">GitHub</a>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
