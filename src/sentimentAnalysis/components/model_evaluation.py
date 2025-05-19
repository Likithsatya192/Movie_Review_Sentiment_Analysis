from typing import Any, Dict
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import pickle

class ModelEvaluation:
    def __init__(self, model_path: str, tokenizer_path: str, X_test: np.ndarray, y_test: np.ndarray):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.X_test = X_test
        self.y_test = y_test
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()

    def _load_model(self):
        return tf.keras.models.load_model(self.model_path)

    def _load_tokenizer(self):
        with open(self.tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        return tokenizer

    def evaluate(self) -> Dict[str, Any]:
        # Ensure input shape matches model expectation
        expected_len = self.model.input_shape[1]
        if self.X_test.shape[1] != expected_len:
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            self.X_test = pad_sequences(self.X_test, maxlen=expected_len, padding='post', truncating='post')
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        y_pred_prob = self.model.predict(self.X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        report = classification_report(self.y_test, y_pred, output_dict=True)
        cm = confusion_matrix(self.y_test, y_pred)
        return {
            'loss': loss,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
