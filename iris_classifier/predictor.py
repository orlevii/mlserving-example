import os

import joblib
import numpy as np
from mest.api import request_schema
from mest.predictors import RESTPredictor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_array

from config import config
from .trainer import FEATURES_TO_NORMALIZE, FEATURES_NAMES, MODEL_FILE_NAME, NORMALIZER_FILE_NAME

REQUEST_SCHEMA = {
    'sepal_length': 'float',
    'sepal_width': 'float',
    'petal_length': 'float',
    'petal_width': 'float',
}


@request_schema(REQUEST_SCHEMA)
class IrisClassifierPredictor(RESTPredictor):

    def __init__(self):
        # Load classifier
        classifier_path = os.path.join(config.LOCAL_MODEL_DIR, MODEL_FILE_NAME)
        self.classifier: LogisticRegression = joblib.load(classifier_path)

        # Load normalizer
        normalizer_path = os.path.join(config.LOCAL_MODEL_DIR, NORMALIZER_FILE_NAME)
        self.normalizer: MinMaxScaler = joblib.load(normalizer_path)

    def pre_process(self, features: dict, req):
        """
        Note - the pre-processing is not using pandas DataFrame for performance reasons.
        Using numpy might require more code, but the performance is MUCH faster.
        """
        for feature_name, value in features.items():
            features[feature_name] = [value]

        self.normalize_features(features)
        # Sort features in the right order
        return check_array([features[name] for name in FEATURES_NAMES]).transpose()

    def predict(self, processed_data: np.ndarray, req):
        return self.classifier.predict_proba(processed_data)[0]

    def post_process(self, prediction, req) -> dict:
        return {
            'probabilities': {
                'setosa': prediction[0],
                'versicolour': prediction[1],
                'virginica': prediction[2]
            }
        }

    def normalize_features(self, features: dict):
        # Get features to normalize
        matrix = [features[feature_name] for feature_name in FEATURES_TO_NORMALIZE]
        matrix = check_array(matrix).transpose()

        # Normalize
        normalized_matrix = self.normalizer.transform(matrix).transpose()

        # Update the features dict
        for i in range(len(FEATURES_TO_NORMALIZE)):
            feature_name = FEATURES_TO_NORMALIZE[i]
            vector = normalized_matrix[i]

            features[feature_name] = vector
