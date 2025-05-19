import os
import pickle
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH = 'artifacts/model_trainer/model.h5'
TOKENIZER_PATH = 'artifacts/model_trainer/tokenizer.pickle'
MAX_LEN = 200  # Should match your config/params

app = Flask(__name__, static_folder='static')

# Load model and tokenizer at startup
model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    prob = model.predict(pad)[0][0]
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
    app.run(debug=True)
