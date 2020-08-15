import os

import joblib
import pandas as pd
from mest.predictors import RESTPredictor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from config import config
from .trainer import FEATURES_TO_NORMALIZE, FEATURES_NAMES, MODEL_FILE_NAME, NORMALIZER_FILE_NAME


class IrisClassifierPredictor(RESTPredictor):
    REQUEST_SCHEMA = {
        'sepal_length': {'type': 'float', 'required': True},
        'sepal_width': {'type': 'float', 'required': True},
        'petal_length': {'type': 'float', 'required': True},
        'petal_width': {'type': 'float', 'required': True},
    }

    def __init__(self):
        # Load classifier
        classifier_path = os.path.join(config.LOCAL_MODEL_DIR, MODEL_FILE_NAME)
        self.classifier: LogisticRegression = joblib.load(classifier_path)

        # Load normalizer
        normalizer_path = os.path.join(config.LOCAL_MODEL_DIR, NORMALIZER_FILE_NAME)
        self.normalizer: MinMaxScaler = joblib.load(normalizer_path)

    def pre_process(self, features: dict, req):
        for feature_name, value in features.items():
            features[feature_name] = [value]

        # Create dataframe, select the features in the right order.
        features_df = pd.DataFrame(features)[FEATURES_NAMES]
        self.normalize_features(features_df)

        return features_df

    def predict(self, processed_data: pd.DataFrame, req):
        return self.classifier.predict_proba(processed_data)[0]

    def post_process(self, prediction, req) -> dict:
        return {
            'probabilities': {
                'setosa': prediction[0],
                'versicolour': prediction[1],
                'virginica': prediction[2]
            }
        }

    def normalize_features(self, features: pd.DataFrame):
        # Use normalizer.transform(...) on the features we want to normalize
        features[FEATURES_TO_NORMALIZE] = self.normalizer.transform(features[FEATURES_TO_NORMALIZE])
