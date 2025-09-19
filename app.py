# app.py
import streamlit as st
import pickle
import time
import os



# Load Model and Vectorizer (Safe Path)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================

model = pickle.load(open(os.path.join(BASE_DIR, 'spam_model.pkl'), 'rb'))
vectorizer = pickle.load(open(os.path.join(BASE_DIR, 'vectorizer.pkl'), 'rb'))

# ==============================
# Prediction Function
# ==============================
def predict_spam(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    prob = model.predict_proba(text_vec)[0]  # probability
    return prediction, prob


# ==============================
# Streamlit App
# ==============================
def main():
    st.set_page_config(page_title="Spam or Ham Classifier", page_icon="📧", layout="centered")

    st.title("📧 Spam or Ham Classifier")
    st.markdown("### Detect whether a message is *Spam* or *Ham* using Machine Learning.")

    # Sidebar
    st.sidebar.header("ℹ About the App")
    st.sidebar.write(
        """
        This app uses a *Naive Bayes Classifier* trained on the SMS Spam dataset.  
        Enter a message in the box and click *Predict* to see the result.  
        """
    )
    st.sidebar.write("📌 Built with *Streamlit, Scikit-learn, Python*")

    # Input area
    st.markdown("#### ✍ Enter your message below:")
    user_input = st.text_area("Message:", height=150, placeholder="Type your SMS or Email here...")

    # Predict button
    if st.button("🔍 Predict"):
        if user_input.strip() == "":
            st.warning("⚠ Please enter a message first!")
        else:
            with st.spinner("Analyzing message..."):
                time.sleep(1)  # just to show spinner
                result, prob = predict_spam(user_input)

            # Display result
            if result == 1:
                st.error("🚨 This message is *Spam*!")
            else:
                st.success("✅ This message is *Ham* (Not Spam).")

            # Show probabilities
            st.markdown("### 📊 Prediction Probability")
            st.progress(float(max(prob)))  # progress bar
            st.write(f"🔹 Probability Ham: *{prob[0]:.4f}*")
            st.write(f"🔸 Probability Spam: *{prob[1]:.4f}*")

    # Footer
    st.markdown("---")
    st.markdown("Made with ❤ using Streamlit | Demo Project")


# Run App
if _name_ == "_main_":
    main()
