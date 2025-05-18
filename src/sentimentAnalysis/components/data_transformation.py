import os
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sentimentAnalysis.logging import logger

class DataTransformation:
    def __init__(self, config):
        self.config = config

    def map_labels(self, series):
        label_map = {'negative': 0, 'positive': 1, 'neg': 0, 'pos': 1, 0: 0, 1: 1}
        return series.map(lambda x: label_map.get(str(x).lower(), x)).astype(int)

    def run(self):
        logger.info("Loading training and test data for transformation...")
        train_df = pd.read_csv(self.config.train_data_path)
        test_df = pd.read_csv(self.config.test_data_path)

        train_texts = train_df['text'].astype(str).tolist()
        test_texts = test_df['text'].astype(str).tolist()
        train_labels = self.map_labels(train_df['sentiment'])
        test_labels = self.map_labels(test_df['sentiment'])

        logger.info("Fitting tokenizer on training data...")
        tokenizer = Tokenizer(num_words=self.config.max_length, oov_token="<OOV>")
        tokenizer.fit_on_texts(train_texts)

        logger.info("Converting texts to sequences...")
        X_train_seq = tokenizer.texts_to_sequences(train_texts)
        X_test_seq = tokenizer.texts_to_sequences(test_texts)

        logger.info("Padding sequences to max length %d...", self.config.max_length)
        X_train_pad = pad_sequences(X_train_seq, maxlen=self.config.max_length, padding='post', truncating='post')
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.config.max_length, padding='post', truncating='post')

        logger.info("Saving processed data and tokenizer to %s...", self.config.preprocessing_output)
        os.makedirs(self.config.preprocessing_output, exist_ok=True)
        np.savez_compressed(os.path.join(self.config.preprocessing_output, "train.npz"),
                            input_ids=X_train_pad, labels=train_labels)
        np.savez_compressed(os.path.join(self.config.preprocessing_output, "test.npz"),
                            input_ids=X_test_pad, labels=test_labels)
        with open(os.path.join(self.config.preprocessing_output, "tokenizer.pickle"), "wb") as f:
            pickle.dump(tokenizer, f)

        logger.info("Data transformation complete. Processed files and tokenizer saved to: %s", self.config.preprocessing_output) 