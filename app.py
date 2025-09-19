# app.py
import streamlit as st
import joblib
import os

# ==============================
# Load Model and Vectorizer
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "models", "spam_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "models", "vectorizer.pkl"))

# ==============================
# Prediction Function
# ==============================
def predict_spam_or_ham(message):
    message_vectorized = vectorizer.transform([message])
    prediction = model.predict(message_vectorized)

    if prediction == 1:
        return "Spam"
    else:
        return "Ham"

# ==============================
# Streamlit App UI
# ==============================
st.title("SMS Spam Detection")
st.header("Enter the message to classify")

message = st.text_area("Message", "Type here...")

if st.button("Predict"):
    if message.strip():
        result = predict_spam_or_ham(message)
        st.success(f"This message is: {result}")
    else:
        st.warning("Please enter a message to classify")
