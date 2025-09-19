# app.py
import streamlit as st
import joblib
import time
import os

# ==============================
# Load Model and Vectorizer (Safe Path)
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath('/content/spam_model.pkl'))
BASE_DIR = os.path.dirname(os.path.abspath('/content/vectorizer.pkl'))
model = joblib.load(os.path.join(BASE_DIR, "spam_model.pkl"),"rb")
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"),"rb")

# ==============================
# Prediction Function
# ==============================
def predict_spam(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    prob = model.predict_proba(text_vec)[0]
    return prediction, prob

# ==============================
# Streamlit App
# ==============================
def main():
    st.set_page_config(page_title="Spam or Ham Classifier", page_icon="📧", layout="centered")
    st.title("📧 Spam or Ham Classifier")

    user_input = st.text_area("✍ Enter your message below:", height=150)

    if st.button("🔍 Predict"):
        if user_input.strip() == "":
            st.warning("⚠ Please enter a message first!")
        else:
            with st.spinner("Analyzing message..."):
                time.sleep(1)
                result, prob = predict_spam(user_input)

            if result == 1:
                st.error("🚨 Spam!")
            else:
                st.success("✅ Ham!")

            st.write(f"🔹 Probability Ham: {prob[0]:.4f}")
            st.write(f"🔸 Probability Spam: {prob[1]:.4f}")

if _name_ == "_main_":
    main()
