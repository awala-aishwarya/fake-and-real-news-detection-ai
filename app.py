import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')

st.title("📰 Fake News Detection App")  # <- this must be present

def clean_text(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Load the saved model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

news_text = st.text_area("Enter News Article Text:")

if st.button("Predict"):
    if news_text:
        cleaned = clean_text(news_text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        result = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
        st.success(f"Prediction: {result}")
    else:
        st.warning("Please enter some news text.")
