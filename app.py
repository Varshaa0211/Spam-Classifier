import streamlit as st
import joblib

# ----------------------------
# Load Model and Vectorizer
# ----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="Spam Classification", layout="centered")
st.title("📩 Spam Message Classifier")

st.write("Type a message below to check whether it is **Spam** or **Ham (Not Spam)**.")

# User Input
user_input = st.text_area("Enter your message:", "")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message first!")
    else:
        # Transform input and predict
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]

        if prediction == 1:
            st.error("🚨 This message is **SPAM**!")
        else:
            st.success("✅ This message is **HAM (Not Spam)**.")
