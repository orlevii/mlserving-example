from utils.model_version import ModelVersion

model_version = ModelVersion.find_or_create('iris_classifier')

model_version.bump('./model_artifacts')

model_version.get('./test')
