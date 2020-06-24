import os

import joblib
from mest.models import BaseModel
from mest.predictors import BasePredictor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from config import config
from utils.model_version import ModelVersion
from .predictor import IrisClassifierPredictor
from .trainer import MODEL_FILE_NAME, NORMALIZER_FILE_NAME


class IrisClassifierModel(BaseModel):
    def __init__(self):


        self.classifier: LogisticRegression = joblib.load(os.path.join(config.LOCAL_MODEL_DIR, MODEL_FILE_NAME))
        self.normalizer: MinMaxScaler = joblib.load(os.path.join(config.LOCAL_MODEL_DIR, NORMALIZER_FILE_NAME))

    def create_predictor(self) -> BasePredictor:
        return IrisClassifierPredictor(self)
