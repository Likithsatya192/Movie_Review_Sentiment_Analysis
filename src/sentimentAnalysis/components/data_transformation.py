import os
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sentimentAnalysis.logging import logger

class DataTransformation:
    def __init__(self, config):
        self.config = config

    def map_labels(self, series):
        label_map = {'negative': 0, 'positive': 1, 'neg': 0, 'pos': 1, 0: 0, 1: 1}
        return series.map(lambda x: label_map.get(str(x).lower(), x)).astype(int)

    def run(self):
        logger.info("Loading data for transformation...")
        df = pd.read_csv(self.config.train_data_path)
        texts = df[self.config.text_column].astype(str).tolist()
        labels = np.array(self.map_labels(df[self.config.label_column]), dtype=np.float32)
        assert not np.any(pd.isnull(labels)), "NaN in labels"

        logger.info("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=self.config.test_size, random_state=42, stratify=labels
        )

        logger.info("Fitting tokenizer on training data...")
        tokenizer = Tokenizer(num_words=self.config.max_features, oov_token="<OOV>")
        tokenizer.fit_on_texts(X_train)

        logger.info("Converting texts to sequences...")
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)

        logger.info("Padding sequences to max length %d...", self.config.max_len)
        X_train_pad = pad_sequences(X_train_seq, maxlen=self.config.max_len, padding='post', truncating='post')
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.config.max_len, padding='post', truncating='post')

        logger.info("Saving processed data and tokenizer to %s...", self.config.preprocessing_output)
        os.makedirs(self.config.preprocessing_output, exist_ok=True)
        np.savez_compressed(os.path.join(self.config.preprocessing_output, "train.npz"),
                            input_ids=X_train_pad, labels=y_train)
        np.savez_compressed(os.path.join(self.config.preprocessing_output, "test.npz"),
                            input_ids=X_test_pad, labels=y_test)
        with open(os.path.join(self.config.preprocessing_output, "tokenizer.pickle"), "wb") as f:
            pickle.dump(tokenizer, f)

        logger.info("Data transformation complete. Processed files and tokenizer saved to: %s", self.config.preprocessing_output)