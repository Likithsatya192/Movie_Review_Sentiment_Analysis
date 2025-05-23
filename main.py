from sentimentAnalysis.logging import logger
from sentimentAnalysis.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from sentimentAnalysis.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from sentimentAnalysis.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from sentimentAnalysis.pipeline.stage_04_model_evaluation import ModelEvaluationTrainingPipeline


STAGE_NAME = "Data Ingestion stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Validation stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_validation = DataValidationTrainingPipeline()
    data_validation.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Transformation stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME = "Model Evaluation stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_evaluation = ModelEvaluationTrainingPipeline()
    model_evaluation.run()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
except Exception as e:
    logger.exception(e)
    raise e