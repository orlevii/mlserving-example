from typing import Any

from mest.api import Response
from mest.predictors import BasePredictor
from sklearn.utils import check_array

from .trainer import FEATURES_TO_NORMALIZE, FEATURES_NAMES


class IrisClassifierPredictor(BasePredictor):
    REQUEST_SCHEMA = {
        'sepal_length': {'type': 'float', 'required': True},
        'sepal_width': {'type': 'float', 'required': True},
        'petal_length': {'type': 'float', 'required': True},
        'petal_width': {'type': 'float', 'required': True},
    }

    def __init__(self, model):
        super().__init__(model)

    def pre_process(self, features: dict):
        # make a vector from each value
        for feature_name, value in features.items():
            features[feature_name] = [value]

        self.normalize_features(features)

        # Sort features in the right order
        return check_array([features[name] for name in FEATURES_NAMES]).transpose()

    def infer(self, processed_data: Any):
        return self.model.classifier.predict_proba(processed_data)[0]

    def post_process(self, prediction) -> Response:
        return Response(
            data={
                'probabilities': {
                    'setosa': prediction[0],
                    'versicolour': prediction[1],
                    'virginica': prediction[2]
                }
            }
        )

    def normalize_features(self, features: dict):
        matrix = [features[feature_name] for feature_name in FEATURES_TO_NORMALIZE]
        matrix = check_array(matrix).transpose()

        normalized_matrix = self.model.normalizer.transform(matrix).transpose()

        for i in range(len(FEATURES_TO_NORMALIZE)):
            feature_name = FEATURES_TO_NORMALIZE[i]
            vector = normalized_matrix[i]

            features[feature_name] = vector
