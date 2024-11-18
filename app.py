from flask import Flask, request, jsonify
import nltk
from keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import preprocess_new_text, get_preprocessed_dataset
import joblib
import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

app = Flask(__name__)

dataPath = "./data"
categories = ["Crime", "Entertainment", "Politics", "Science"]
tfidfPath = './models/tfidf_vectorizer.pkl'

if os.path.exists(tfidfPath):
    tfidf = joblib.load(tfidfPath)
else:
    balancedDf, yResampled = get_preprocessed_dataset(dataPath, categories)
    tfidf = TfidfVectorizer(max_features=5000, lowercase=False)
    tfidf.fit_transform(balancedDf)
    joblib.dump(tfidf, tfidfPath)

model = None

def getModel():
    global model
    if model is None:
        model = load_model("./models/w2v_nn.keras")
    return model

# Endpoint for classification
@app.route('/classify', methods=['POST'])
def classifyEmail():
    try:
        data = request.get_json()
        emailText = data.get("emailText", "")
        if not emailText:
            return jsonify({"error": "No emailText provided"}), 400

        vectorizedText = preprocess_new_text(emailText, tfidf)
        prediction = getModel().predict(vectorizedText)
        labels = ["Crime", "Entertainment", "Politics", "Science"]
        predictedCategory = labels[prediction.argmax()]

        return jsonify({"category": predictedCategory}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
