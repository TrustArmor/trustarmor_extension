import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import re
import numpy as np

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, urls):
        features = []
        for url in urls:
            feature = []
            feature.append(len(url)) 
            feature.append(len(re.findall(r'[^a-zA-Z0-9]', url)))
            feature.append(len(url.split('.')) - 2) 
            suspicious_keywords = ['free', 'download', 'click', 'win', 'prize', 'example']
            feature.append(sum(1 for keyword in suspicious_keywords if keyword in url.lower())) 
            features.append(feature)
        return np.array(features)

def train_model():
    data = pd.read_csv("balanced_urls.csv")

    data['label'] = data['label'].map({
        'benign': 0,
        'malicious': 1
    })
    data['label'].fillna(0, inplace=True)

    x = data['url']
    y = data['label']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=32)

    vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=5000)
    tfidf_transformer = TfidfTransformer()
    x_train_counts = vectorizer.fit_transform(x_train)
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

    model = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', random_state=42)
    model.fit(x_train_tfidf, y_train)

    x_test_counts = vectorizer.transform(x_test)
    x_test_tfidf = tfidf_transformer.transform(x_test_counts)
    y_pred = model.predict(x_test_tfidf)

    print(classification_report(y_test, y_pred))

    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(conf_matrix)

    joblib.dump(model, 'url_classifier.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    joblib.dump(tfidf_transformer, 'tfidf_transformer.pkl')

if __name__ == '__main__':
    train_model()
