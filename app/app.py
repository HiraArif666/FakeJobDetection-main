from pathlib import Path
import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Streamlit config should be the first Streamlit call
st.set_page_config(page_title="Fake Job Post Detector", page_icon="🛰️", layout="centered")

# Resolve paths relative to this script (app/app.py)
BASE_DIR = Path(__file__).resolve().parent

# =============================
# LOAD MODELS AND TOKENIZERS
# =============================
@st.cache_resource
def load_models():
    # Load TF-IDF model + vectorizer
    with open(BASE_DIR / "tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vectorizer = pickle.load(f)
    with open(BASE_DIR / "model_tfidf_lr.pkl", "rb") as f:
        lr_model = pickle.load(f)

    # Load Bi-LSTM model + tokenizer
    bilstm_model = tf.keras.models.load_model(BASE_DIR / "model_bilstm.h5", compile=False)
    with open(BASE_DIR / "tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    return tfidf_vectorizer, lr_model, bilstm_model, tokenizer

tfidf_vectorizer, lr_model, bilstm_model, tokenizer = load_models()

# =============================
# STREAMLIT UI
# =============================
st.title("🚀 Deep Learning–Based Fake Job Post Detection")
st.write("Paste a job posting below to check if it's **Real** or **Fake** using AI models.")

# User input
user_input = st.text_area("Enter job description:", height=200, placeholder="Type or paste a job description here...")

# Prediction button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # TF-IDF Prediction
        tfidf_features = tfidf_vectorizer.transform([user_input])
        lr_pred = int(lr_model.predict(tfidf_features)[0])

        # Bi-LSTM Prediction
        seq = tokenizer.texts_to_sequences([user_input])
        pad = pad_sequences(seq, maxlen=200, padding='post')
        bilstm_pred = bilstm_model.predict(pad, verbose=0)
        bilstm_pred_label = int((bilstm_pred > 0.5).astype(int)[0][0])

        # Display results
        st.subheader("🧠 Model Predictions")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("TF-IDF + Logistic Regression", "Fake" if lr_pred == 1 else "Real")

        with col2:
            st.metric("Bi-LSTM Deep Learning", "Fake" if bilstm_pred_label == 1 else "Real")

        # Combined verdict
        if lr_pred == 1 or bilstm_pred_label == 1:
            st.error("🚨 This job post is likely **FAKE**. Be cautious!")
        else:
            st.success("✅ This job post appears **REAL**. It seems legitimate.")

st.caption("Developed as part of Buildables Data Science Fellowship – Fake Job Post Detection (2025)")