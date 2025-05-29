# fake_news_detection

This project is a comprehensive Machine Learning system that automatically detects fake news and categorizes it based on content. It uses a combination of traditional and modern techniques including Naive Bayes, Logistic Regression, Artificial Neural Networks (ANN), TF-IDF feature extraction, Cosine Similarity, Clustering, and Reinforcement Learning for adaptive accuracy improvement.

---

## 🚀 Features

- Detects whether a news article is **Fake** or **Real**
- Highlights **suspicious keywords** influencing the prediction
- Supports **multiple ML models**: Naive Bayes, Logistic Regression, ANN
- Calculates **Cosine Similarity** to detect plagiarism or duplicates
- Categorizes news into topics using **KMeans Clustering**
- Learns from feedback using simulated **Reinforcement Learning**
- Includes robust **text preprocessing** (cleaning, lemmatization, TF-IDF)

---

## 📂 Project Structure
fake_news_detection/
├── data/ # Sample dataset (CSV)
├── models/ # ML model definitions
├── utils/ # Preprocessing and TF-IDF tools
├── clustering/ # Topic categorization via clustering
├── rl/ # Simulated feedback loop
├── predict.py # Interactive CLI-based fake news predictor
├── main.py # Core pipeline (training + clustering + feedback)
├── requirements.txt
└── README.md

# sample output

Enter news title: Miracle COVID Cure Found
Enter news article content: Scientists discovered a miracle cure using alien plants...

Prediction: ❌ FAKE
Confidence - Fake: 0.95, Real: 0.05
Top contributing words: ['miracle', 'cure', 'alien']
