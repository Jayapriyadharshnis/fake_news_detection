import streamlit as st
import pandas as pd
import re, string, warnings
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline
warnings.filterwarnings('ignore')

st.set_page_config(page_title='Fake News Detector',
    page_icon='🔍', layout='wide')

@st.cache_resource
def load_and_train():
    df = pd.read_csv('fake_or_real_news.csv')
    df.drop(columns=['Unnamed: 0'], inplace=True)

    def clean(text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('','',string.punctuation))
        return re.sub(r'\s+', ' ', text).strip()

    df['clean'] = (df['title'] + ' ' + df['text']).apply(clean)
    df['label_enc'] = df['label'].map({'REAL':1,'FAKE':0})

    X_train,X_test,y_train,y_test = train_test_split(
        df['clean'], df['label_enc'],
        test_size=0.2, random_state=42, stratify=df['label_enc'])

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100)
    }
    trained = {}
    for name, clf in models.items():
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000,
                      ngram_range=(1,2), stop_words='english')),
            ('clf', clf)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        trained[name] = {'pipe':pipe,
            'acc':accuracy_score(y_test,y_pred), 'pred':y_pred}
    return df, trained, X_test, y_test, clean

df, trained, X_test, y_test, clean_fn = load_and_train()
best = max(trained, key=lambda k: trained[k]['acc'])
best_pipe = trained[best]['pipe']

st.title('🔍 Fake News Detector')
tab1, tab2, tab3 = st.tabs(['Predict','Model Performance','Word Clouds'])

with tab1:
    user_input = st.text_area('Paste news article:', height=200)
    if st.button('Analyze'):
        if user_input.strip():
            pred = best_pipe.predict([clean_fn(user_input)])[0]
            proba = best_pipe.predict_proba([clean_fn(user_input)])[0]
            label = 'REAL' if pred==1 else 'FAKE'
            st.success(f'{label} — {max(proba)*100:.1f}% confidence')
            col1,col2 = st.columns(2)
            col1.metric('FAKE %', f'{proba[0]*100:.1f}%')
            col2.metric('REAL %', f'{proba[1]*100:.1f}%')

with tab2:
    for name, data in trained.items():
        st.metric(name, f"{data['acc']*100:.2f}%")
    cm = confusion_matrix(y_test, trained[best]['pred'])
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm,display_labels=['FAKE','REAL']).plot(ax=ax)
    st.pyplot(fig)

with tab3:
    fake_text = ' '.join(df[df['label']=='FAKE']['clean'])
    real_text = ' '.join(df[df['label']=='REAL']['clean'])
    col1,col2 = st.columns(2)
    with col1:
        st.markdown('### FAKE News Words')
        wc = WordCloud(width=600,height=350,
                 background_color='white',colormap='Reds').generate(fake_text)
        fig1,ax1 = plt.subplots()
        ax1.imshow(wc); ax1.axis('off')
        st.pyplot(fig1)
    with col2:
        st.markdown('### REAL News Words')
        wc = WordCloud(width=600,height=350,
                 background_color='white',colormap='Greens').generate(real_text)
        fig2,ax2 = plt.subplots()
        ax2.imshow(wc); ax2.axis('off')
        st.pyplot(fig2)
