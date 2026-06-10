# fake_news_detection
Fake News Detection using NLP and Machine Learning — TF-IDF + Logistic Regression achieving 93.97% accuracy with Streamlit web app
# 🔍 Fake News Detection

A machine learning project that classifies news articles as FAKE or REAL using Natural Language Processing.

## 📊 Results
| Model | Accuracy |
|---|---|
| Logistic Regression | 93.97% ✅ Best |
| Random Forest | 91.31% |
| Naive Bayes | 89.98% |

## 🛠️ Tech Stack
- Python
- Scikit-learn
- TF-IDF Vectorization
- Streamlit
- Matplotlib
- WordCloud

## ✨ Features
- Classifies news as FAKE or REAL in real time
- Compares 3 ML models
- Interactive Streamlit web app
- Word cloud visualization for FAKE vs REAL news
- Confusion matrix and classification report

## 📁 Dataset
- 6385 labeled news articles (FAKE + REAL)
- Source: fake_or_real_news.csv
- Download dataset from Kaggle: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Place it in the project folder and rename it to fake_or_real_news.csv

## 🚀 Run Locally
pip install -r requirements.txt
streamlit run app.py

## 📌 Project Structure
fake_news_detection/
├── fake_or_real_news.csv
├── fake_news_detection.py
├── app.py
├── requirements.txt
├── README.md
└── outputs/
    ├── confusion_matrix.png
    └── model_comparison.png
