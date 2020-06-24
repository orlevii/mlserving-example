from mest import Mest

from iris_classifier.model import IrisClassifierModel

mest = Mest()

mest.add_inference_handler(IrisClassifierModel(), '/api/v1/predict')
app = mest.app

if __name__ == '__main__':
    mest.run()

"""
curl -X POST http://127.0.0.1:5000/api/v1/predict \
-H 'Content-Type: application/json' -d '{
    "sepal_length": 0.1,
    "sepal_width": 0.2,
    "petal_length": 0.3,
    "petal_width": 0.4
}'
"""
