import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.preprocess import clean_text
import joblib

# Load dataset
df = pd.read_csv("data/sample_news.csv")
df["clean_text"] = df["text"].apply(clean_text)

# Vectorize
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

# Train model (you can also load pre-trained with joblib if needed)
model = MultinomialNB()
model.fit(X, y)

# Get input from user
print("=== Fake News Detector ===")
user_title = input("Enter news title: ")
user_text = input("Enter news article content: ")

# Preprocess and vectorize input
full_text = user_title + " " + user_text
cleaned = clean_text(full_text)
user_vec = vectorizer.transform([cleaned])

# Prediction
prediction = model.predict(user_vec)[0]
probs = model.predict_proba(user_vec)[0]

# Explanation (top suspicious words)
feature_names = vectorizer.get_feature_names_out()
top_indices = user_vec.toarray()[0].argsort()[-5:][::-1]
top_words = [feature_names[i] for i in top_indices if user_vec.toarray()[0][i] > 0]

# Output
print(f"\nPrediction: {'❌ FAKE' if prediction == 'fake' else '✅ REAL'}")
print(f"Confidence - Fake: {probs[model.classes_ == 'fake'][0]:.2f}, Real: {probs[model.classes_ == 'real'][0]:.2f}")
print("Top contributing words:", top_words)
