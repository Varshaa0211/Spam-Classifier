import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download only what is needed (no punkt!)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Load stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load the trained model, vectorizer, and label encoder
try:
    model = joblib.load('logistic_regression_sms_spam_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    le = joblib.load('label_encoder.pkl')
except FileNotFoundError:
    st.error("Model, vectorizer, or label encoder files not found. Please ensure they are saved in the correct location.")
    st.stop()

# Preprocessing function
def preprocess_sms(sms_text):
    # Remove special characters and numbers
    cleaned_text = re.sub(r"[^a-zA-Z\s]", "", sms_text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    # Tokenization (simple split → avoids punkt issue)
    tokens = cleaned_text.split()

    # Remove stopwords
    tokens = [word.lower() for word in tokens if word.lower() not in stop_words]

    # Lemmatization
    tokens = [lemmatizer.lemmatize(word, pos="v") for word in tokens]

    return " ".join(tokens)

# Prediction function
def predict_sms(sms_text):
    processed_sms = preprocess_sms(sms_text)
    vectorized_sms = tfidf_vectorizer.transform([processed_sms])
    prediction = model.predict(vectorized_sms)[0]
    return le.inverse_transform([prediction])[0]

# Streamlit UI
st.title("📩 SMS Spam Classifier")

sms_input = st.text_area("Enter the SMS message:")

if st.button("Predict"):
    if sms_input.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        result = predict_sms(sms_input)
        if result == "spam":
            st.error("🚨 This message is SPAM!")
        else:
            st.success("✅ This message is HAM (Not Spam).")
