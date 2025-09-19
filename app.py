import streamlit as st
import pickle


# Title of the app
st.title('SMS Spam Detection')

# Load pre-trained model and vectorizer
model = pickle.load(open('spam_model.pkl', 'rb'))  # Change the model filename if needed
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))  # The vectorizer used during training

# Function to predict spam or ham
def predict_spam_or_ham(message):
    # Transform the input message using the same vectorizer
    message_vectorized = vectorizer.transform([message])
    prediction = model.predict(message_vectorized)
    
    if prediction == 1:
        return 'Spam'
    else:
        return 'Ham'

# App interface
st.header('Enter the message to classify')

# User input
message = st.text_area("Message", "Type here...")

# Predict button
if st.button('Predict'):
    if message:
        result = predict_spam_or_ham(message)
        st.success(f'This message is: {result}')
    else:
        st.warning('Please enter a message to classify')
