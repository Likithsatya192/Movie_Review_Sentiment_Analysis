import pandas as pd
import numpy as np
import json
from sentimentAnalysis.config.configuration import ConfigurationManager
from sentimentAnalysis.components.model_evaluation import ModelEvaluation

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.eval_config = self.config_manager.get_model_evaluation_config()
        self.params = self.config_manager.params
        self.text_column = self.params['base']['text_column']
        self.label_column = self.params['base']['target_column']

    def map_labels(self, series):
        label_map = {'negative': 0, 'positive': 1, 'neg': 0, 'pos': 1, 0: 0, 1: 1}
        return pd.Series([label_map.get(str(x).lower(), x) for x in series]).astype(int)

    def load_test_data(self):
        df = pd.read_csv(self.eval_config.test_data_path)
        X = df[self.text_column].values
        y = self.map_labels(df[self.label_column].values)
        return X, y

    def get_max_len(self):
        # Try both 'max_len' and 'max_length' for compatibility
        preprocessing = self.params['preprocessing']
        if 'max_len' in preprocessing:
            return preprocessing['max_len']
        elif 'max_length' in preprocessing:
            return preprocessing['max_length']
        else:
            raise KeyError("Neither 'max_len' nor 'max_length' found in preprocessing params.")

    def run(self):
        # Load test data
        X_raw, y = self.load_test_data()
        # Load tokenizer
        import pickle
        with open(self.eval_config.tokenization_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        max_len = self.get_max_len()
        X_seq = tokenizer.texts_to_sequences(X_raw)
        X_pad = pad_sequences(X_seq, maxlen=max_len, padding='post', truncating='post')
        # Evaluate model
        evaluator = ModelEvaluation(
            model_path=self.eval_config.model_path,
            tokenizer_path=self.eval_config.tokenization_path,
            X_test=X_pad,
            y_test=y
        )
        metrics = evaluator.evaluate()
        # Save metrics
        with open(self.eval_config.metric_file_name, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Evaluation metrics saved to {self.eval_config.metric_file_name}")

if __name__ == "__main__":
    pipeline = ModelEvaluationTrainingPipeline()
    pipeline.run()
