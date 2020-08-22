from mlserving import ServingApp

from iris_classifier.predictor import IrisClassifierPredictor

app = ServingApp()

app.add_inference_handler('/api/v1/predict', IrisClassifierPredictor())

if __name__ == '__main__':
    app.run()

"""
curl -X POST http://127.0.0.1:5000/api/v1/predict \
-H 'Content-Type: application/json' -d '{
    "sepal_length": 0.1,
    "sepal_width": 0.2,
    "petal_length": 0.3,
    "petal_width": 0.4
}'
"""
