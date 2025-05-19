from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    data_file: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: List[str]

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_name: str
    max_features: int
    max_len: int
    test_size: float
    text_column: str
    label_column: str
    train_data_path: Path
    preprocessing_output: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    embedding_dim: int
    model_type: str
    batch_size: int
    epochs: int
    learning_rate: float
    model_output_path: Path
    tokenizer_output_path: Path

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    tokenization_path: Path
    model_path: Path
    metric_file_name: Path