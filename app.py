import streamlit as st
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ----------------------------
# NLTK setup
# ----------------------------
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ----------------------------
# Text Preprocessing
# ----------------------------
def preprocess_text(text):
    # lowercase
    text = text.lower()
    # remove numbers
    text = re.sub(r'\d+', '', text)
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # tokenize
    words = text.split()
    # remove stopwords + lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# ----------------------------
# Load Model & Vectorizer
# ----------------------------
model = joblib.load("spam_classifier_model.pkl")   # trained model
vectorizer = joblib.load("vectorizer.pkl")         # TF-IDF or CountVectorizer

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="📧 Spam Classification", layout="centered")

st.title("📧 Spam or Ham Classifier")
st.write("Enter a message/email text below and find out if it's **Spam** or **Ham (Not Spam)**.")

# Input box
user_input = st.text_area("✍️ Type your message here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to classify.")
    else:
        # preprocess
        processed_text = preprocess_text(user_input)
        # vectorize
        vectorized_text = vectorizer.transform([processed_text])
        # predict
        prediction = model.predict(vectorized_text)[0]

        if prediction == 1:  # Spam label
            st.error("🚨 This message is **SPAM**")
        else:  # Ham label
            st.success("✅ This message is **HAM (Not Spam)**")
