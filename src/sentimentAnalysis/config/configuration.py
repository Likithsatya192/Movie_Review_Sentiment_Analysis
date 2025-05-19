from sentimentAnalysis.constants import *
from sentimentAnalysis.utils.common import read_yaml, create_directories
from sentimentAnalysis.entity import (
    DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig
)

class ConfigurationManager:
    def __init__(
            self,
            config_filepath = CONFIG_FILE_PATH,
            params_filepath = PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
        create_directories([self.config.data_transformation.root_dir])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
            data_file=config.data_file
        )
        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        create_directories([config.root_dir])
        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES
        )
        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])
        return DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            tokenizer_name=config.tokenizer_name,
            max_features=config.max_features,
            max_len=config.max_len,
            test_size=config.test_size,
            text_column=config.text_column,
            label_column=config.label_column,
            train_data_path=config.train_data_path,
            preprocessing_output=config.preprocessing_output,
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        create_directories([config.root_dir])
        return ModelTrainerConfig(
            root_dir=config.root_dir,
            embedding_dim=config.embedding_dim,
            model_type=config.model_type,
            batch_size=config.batch_size,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            model_output_path=config.model_output_path,
            tokenizer_output_path=config.tokenizer_output_path
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        create_directories([config.root_dir])
        return ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            tokenization_path=config.tokenization_path,
            model_path=config.model_path,
            metric_file_name=config.metric_file_name
        )
