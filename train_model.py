import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import chardet  

def detect_encoding(file_path, num_bytes=100000):
    with open(file_path, 'rb') as f:
        rawdata = f.read(num_bytes)
    result = chardet.detect(rawdata)
    return result['encoding']

def train_model():
    file_path = "balanced_urls.csv"
    
    
    try:
        data = pd.read_csv(file_path, encoding='latin1') 
        print("Successfully read CSV with 'latin1' encoding.")
    except UnicodeDecodeError as e:
        print(f"UnicodeDecodeError with 'latin1' encoding: {e}")
        print("Attempting to detect encoding using chardet...")
        encoding = detect_encoding(file_path)
        print(f"Detected encoding: {encoding}")
        data = pd.read_csv(file_path, encoding=encoding)
    
    # Option 2: Automatically detect encoding (uncomment if preferred)
    # encoding = detect_encoding(file_path)
    # print(f"Detected encoding: {encoding}")
    # data = pd.read_csv(file_path, encoding=encoding)

    # Map labels to numerical values
    data['label'] = data['label'].map({
        'benign': 0,
        # 'defacement': 1,
        # 'phishing': 2,
        'malicious': 1,
    })
    data['label'].fillna(0, inplace=True) 

    x = data['url']
    y = data['label']

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=32, stratify=y
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
    x_train_vectorized = vectorizer.fit_transform(x_train)
    x_test_vectorized = vectorizer.transform(x_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight='balanced',
        random_state=42
    )
    model.fit(x_train_vectorized, y_train)

    y_pred = model.predict(x_test_vectorized)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, 'url_classifier.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    print("Model and vectorizer have been saved successfully.")

if __name__ == '__main__':
    train_model()
