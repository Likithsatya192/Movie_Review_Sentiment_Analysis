# End-to-End Movie Review Sentiment Analysis

A full-stack, production-ready project for sentiment analysis of movie reviews using deep learning (TensorFlow/Keras) and a beautiful Flask web UI.

## 🚀 Features
- **Deep Learning Model**: LSTM-based sentiment classifier trained on IMDB reviews.
- **Modern Web UI**: Responsive, glassmorphic, animated interface (Bootstrap 5, custom CSS).
- **Real-Time Prediction**: Enter a review and instantly see if it's Positive or Negative, with confidence.
- **Example Reviews**: Try demo reviews with one click.
- **Pipeline Automation**: Modular pipeline for data ingestion, validation, transformation, and evaluation.
- **Dockerized**: Run anywhere with a single command.
- **Editable Install**: Modern Python packaging with `src/` layout and `setup.py`.

## 🛠️ Tech Stack
- Python 3.10+
- TensorFlow / Keras
- Flask
- Bootstrap 5, custom CSS
- Docker
- scikit-learn, pandas, numpy, etc.

## 📦 Project Structure
```
Movie_Review_Sentiment_Analysis/
├── app.py                # Flask web app
├── main.py               # Pipeline runner (no model training step)
├── requirements.txt      # All dependencies
├── setup.py              # Python packaging
├── Dockerfile            # For containerization
├── src/
│   └── sentimentAnalysis/  # All backend, pipeline, and model code
├── templates/
│   └── index.html        # Beautiful web UI
├── static/
│   └── style.css         # Custom styles
├── artifacts/            # Model, tokenizer, and pipeline outputs
├── config/               # YAML config files
├── params.yaml           # Model and pipeline parameters
└── ...
```

## ⚡ Quickstart
### 1. Clone the repo
```bash
git clone https://github.com/Likithsatya192/Movie_Review_Sentiment_Analysis
cd Movie_Review_Sentiment_Analysis
```

### 2. Install dependencies (local)
```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Run the pipeline (except model training)
```bash
python main.py
```

### 4. Start the web app
```bash
python app.py
```
Visit [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🐳 Docker
Build and run the app in a container:
```bash
docker build -t movie-review-sentiment .
docker run -p 5000:5000 movie-review-sentiment
```

---

## 📝 Usage
- Enter a movie review in the text box and click **Analyze Sentiment**.
- The app will display **Positive** or **Negative** with a confidence score.
- Try the example reviews for a quick demo.

---

## 🧩 Pipeline Steps
1. **Data Ingestion**: Downloads and prepares the IMDB dataset.
2. **Data Validation**: Checks data integrity.
3. **Data Transformation**: Tokenizes, pads, and splits data.
4. **Model Evaluation**: Evaluates the trained model (model training is skipped in this pipeline).

> **Note:** The model and tokenizer must exist in `artifacts/model_trainer/` for the web app to work.

---

## 🤝 Credits
- [TensorFlow](https://www.tensorflow.org/)
- [Flask](https://flask.palletsprojects.com/)
- [Bootstrap](https://getbootstrap.com/)
- IMDB Dataset

---

## 📬 Contact
For questions, suggestions, or contributions, open an issue or PR on GitHub.