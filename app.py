import os
import pickle
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sentimentAnalysis.logging import logger

MODEL_PATH = 'artifacts/model_trainer/model.h5'
TOKENIZER_PATH = 'artifacts/model_trainer/tokenizer.pickle'
MAX_LEN = 200

app = Flask(__name__, static_folder='static')

model = None
tokenizer = None

def _load_artifacts():
    """Load model and tokenizer into globals if not already loaded."""
    global model, tokenizer
    try:
        if model is None:
            logger.info(f"Loading model from {MODEL_PATH}")
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
            model = load_model(MODEL_PATH)
        if tokenizer is None:
            logger.info(f"Loading tokenizer from {TOKENIZER_PATH}")
            if not os.path.exists(TOKENIZER_PATH):
                raise FileNotFoundError(f"Tokenizer file not found: {TOKENIZER_PATH}")
            with open(TOKENIZER_PATH, 'rb') as handle:
                tokenizer = pickle.load(handle)
    except Exception as e:
        logger.exception("Failed to load artifacts: %s", e)
        raise


def predict_sentiment(text):
    if tokenizer is None or model is None:
        _load_artifacts()

    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    try:
        prob = model.predict(pad)[0][0]
    except Exception as e:
        logger.exception("Model prediction failed: %s", e)
        raise
    sentiment = 'Positive' if prob > 0.5 else 'Negative'
    confidence = round(100 * (prob if prob > 0.5 else 1 - prob), 2)
    return sentiment, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    if request.method == 'POST':
        review = request.form['review']
        prediction, confidence = predict_sentiment(review)
    return render_template('index.html', prediction=prediction, confidence=confidence)

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    prediction, confidence = predict_sentiment(review)
    return render_template('index.html', prediction=prediction, confidence=confidence)

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
