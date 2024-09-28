from flask import Flask, request, jsonify
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from train_model import train_model


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

app = Flask(__name__)

model = joblib.load('url_classifier.pkl')
vectorizer = joblib.load('vectorizer.pkl')
tfidf_transformer = joblib.load('tfidf_transformer.pkl')

logging.basicConfig(level=logging.INFO)

@app.route('/predict', methods=['POST'])
def predict():
    logging.info("Request method: %s", request.method)
    logging.info("Request URL: %s", request.url)
    
    if request.content_type != 'application/json':
        return jsonify({"error": "Content-Type must be application/json"}), 415
    
    data = request.json
    url = data.get('url')

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        url_counts = vectorizer.transform([url])
        url_tfidf = tfidf_transformer.transform(url_counts)
        prediction = model.predict(url_tfidf)
        label_mapping = {
            0: 'benign',
            1: 'malicious'
        }
        result = label_mapping.get(prediction[0], 'unknown')
        return jsonify({'result': result})
    except Exception as e:
        logging.error("Error during prediction: %s", str(e))
        return jsonify({"error": "Internal server error"}), 500

@app.route('/train', methods=['POST'])
def train():
    try:
        train_model()
        return jsonify({'message': 'Model trained successfully'}), 200
    except Exception as e:
        logging.error("Error training model: %s", str(e))
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
