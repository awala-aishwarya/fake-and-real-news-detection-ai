Streamlit app that detects fake news using a Naive Bayes machine learning model.# 📰 Fake News Detection AI Project

This is an AI-based web app that detects whether a news article is real or fake using a machine learning model trained on real-world news articles.

## 🚀 How to Run This Project

### Requirements:
- Python 3.8 or higher

### Install Required Packages:
```bash
pip install pandas scikit-learn streamlit nltk joblib

#Run the app:
streamlit run app.py
Then open your browser to http://localhost:8501

📁 Dataset Used
We used a sampled version of the Kaggle Fake and Real News Dataset.

Due to GitHub file size limits, we only included:

Fake_sample.csv – 100 fake news records

True_sample.csv – 100 real news records

You can replace them with full datasets if needed.

🧠 Model Info
Algorithm: Naive Bayes (MultinomialNB)

Text Features: TF-IDF

Accuracy: ~90% on test data

Interface: Built with Streamlit

📂 Project Files
app.py – Streamlit interface

fakenews.py – model training + prediction logic

Fake_sample.csv, True_sample.csv – small datasets

model.pkl, vectorizer.pkl – saved ML model and TF-IDF transformer

👤 Author
Name: [Awala Aishwarya]

College: [Bhoj Reddy Engineering college for women]

GitHub: [https://github.com/awala-aishwarya/fake-and-real-news-detection-ai]
