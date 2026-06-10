import os
import pandas as pd
import numpy as np
import re, string, warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.utils import compute_sample_weight
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)

# 1. Load Dataset
df = pd.read_csv('fake_or_real_news.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)

print(f"Total rows: {len(df)}")
print(df['label'].value_counts())

# 2. Preprocess Text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return re.sub(r'\s+', ' ', text).strip()

df['clean'] = (df['title'] + ' ' + df['text']).apply(clean_text)
df['label_enc'] = df['label'].map({'REAL': 1, 'FAKE': 0})

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], df['label_enc'],
    test_size=0.2, random_state=42, stratify=df['label_enc']
)

# 4. Sample weights to handle domain imbalance
sample_weights = compute_sample_weight('balanced', y_train)

# 5. Train 3 Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, C=2.0),
    'Naive Bayes': MultinomialNB(alpha=0.1),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
}

results = {}
print("\nTraining models...")
for name, model in models.items():
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=1,
            sublinear_tf=True
        )),
        ('clf', model)
    ])

    # Pass sample weights for models that support it
    if name in ['Logistic Regression', 'Random Forest']:
        pipe.fit(X_train, y_train, clf__sample_weight=sample_weights)
    else:
        pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {'pipe': pipe, 'acc': acc, 'pred': y_pred}
    print(f'{name}: {acc*100:.2f}%')

# 6. Best Model
best = max(results, key=lambda k: results[k]['acc'])
print(f"\nBest Model: {best}")
print(classification_report(y_test, results[best]['pred'],
      target_names=['FAKE', 'REAL']))

# 7. Save Confusion Matrix
cm = confusion_matrix(y_test, results[best]['pred'])
disp = ConfusionMatrixDisplay(cm, display_labels=['FAKE', 'REAL'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap='Blues')
plt.savefig('outputs/confusion_matrix.png', dpi=150)
plt.close()
print("Confusion matrix saved!")

# 8. Model comparison chart
names = list(results.keys())
accs = [results[n]['acc'] * 100 for n in names]
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.barh(names, accs, color=['#4C72B0', '#55A868', '#C44E52'])
ax.set_xlim(80, 100)
ax.set_xlabel('Accuracy (%)')
ax.set_title('Model Comparison')
for bar, acc in zip(bars, accs):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f'{acc:.2f}%', va='center', fontsize=10)
plt.tight_layout()
plt.savefig('outputs/model_comparison.png', dpi=150)
plt.close()
print("Model comparison chart saved!")

# 9. Predict Custom Input
def predict(text):
    cleaned = clean_text(text)
    pred = results[best]['pipe'].predict([cleaned])[0]
    proba = results[best]['pipe'].predict_proba([cleaned])[0]
    return 'REAL' if pred == 1 else 'FAKE', max(proba) * 100

# Test predictions
tests = [
    'Scientists at Johns Hopkins developed a blood test that detects 50 types of cancer before symptoms appear with 99% accuracy in clinical trials',
    'NASA discovers water on Mars surface confirmed by scientists',
    'Government secretly adding chemicals to vaccines to control population whistleblower reveals shocking truth',
    'The Federal Reserve raised interest rates by 25 basis points marking the tenth increase since March 2022',
]

print("\nSample Predictions:")
for t in tests:
    label, conf = predict(t)
    print(f"  {label} ({conf:.1f}%) — {t[:60]}...")
