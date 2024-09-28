from flask import Flask, request, jsonify
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from train_model import train_model

app = Flask(__name__)

model = joblib.load('url_classifier.pkl')
vectorizer = joblib.load('vectorizer.pkl')

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
        url_vectorized = vectorizer.transform([url])
        prediction = model.predict(url_vectorized)
        label_mapping = {
            0: 'benign',
            # 1: 'defacement',
            # 2: 'phishing',
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
