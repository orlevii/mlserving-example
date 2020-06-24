# from utils.model_version import ModelVersion
#
# model_version = ModelVersion.find_or_create('iris_classifier')
#
# model_version.bump('./model_artifacts')
#
# model_version.get('./test')

from iris_classifier.model import IrisClassifierModel

model = IrisClassifierModel()
predictor = model.create_predictor()

res = predictor.predict({
    'sepal_length': 0.1,
    'sepal_width': 0.2,
    'petal_length': 0.3,
    'petal_width': 0.4
})

print(res.data)
