import pandas as pd
import string
import nltk
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download NLTK stopwords (only needed once)
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# Clean the text
def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Load and balance the dataset
def load_data():
    fake = pd.read_csv("Fake_sample.csv")
    true = pd.read_csv("True_sample.csv")

    min_len = min(len(fake), len(true))
    fake = fake.sample(n=min_len, random_state=42)
    true = true.sample(n=min_len, random_state=42)

    fake["label"] = 0
    true["label"] = 1

    data = pd.concat([fake, true]).sample(frac=1, random_state=42)
    data["text"] = data["text"].apply(clean_text)
    return data[["text", "label"]]

# Train and evaluate the model
def train_model():
    data = load_data()
    X = data["text"]
    y = data["label"]

    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Save the model and vectorizer
    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    print("✅ Model and vectorizer saved!")

    # Evaluate the model
    predictions = model.predict(X_test)
    print("\n🔍 Accuracy:", accuracy_score(y_test, predictions))
    print("\n📊 Classification Report:\n", classification_report(y_test, predictions))
    print("\n📉 Confusion Matrix:\n", confusion_matrix(y_test, predictions))

    return model, vectorizer

# Main
if __name__ == "__main__":
    train_model()
