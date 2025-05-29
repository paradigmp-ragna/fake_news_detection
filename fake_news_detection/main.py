import pandas as pd
from utils.preprocess import clean_text, vectorize_text
from models.naive_bayes import train_naive_bayes
from models.logistic_regression import train_logistic_regression
from models.ann_model import train_ann
from utils.tfidf_similarity import compute_similarity
from clustering.topic_clustering import perform_clustering
from rl.feedback_loop import adjust_model_based_on_feedback
from sklearn.metrics import accuracy_score

data = pd.read_csv("data/sample_news.csv")
data["clean_text"] = data["text"].apply(clean_text)
X, vectorizer = vectorize_text(data["clean_text"])
y = data["label"]

# Train models
nb_model = train_naive_bayes(X, y)
lr_model = train_logistic_regression(X, y)
ann_model = train_ann(X, y)

print("=== Model Predictions ===")
nb_preds = nb_model.predict(X)
lr_preds = lr_model.predict(X)
ann_preds = ann_model.predict(X)

print("Naive Bayes Accuracy:", accuracy_score(y, nb_preds))
print("Logistic Regression Accuracy:", accuracy_score(y, lr_preds))
print("ANN Accuracy:", accuracy_score(y, ann_preds))

# Similarity check
similarities = compute_similarity(X)

print("\n=== Cosine Similarity (First 2 Articles) ===")
print(similarities[0][1])

# Clustering
labels = perform_clustering(X)

print("\n=== Clustering Labels ===")
print(labels)

# Simulate feedback loop
adjust_model_based_on_feedback(nb_model, None)
