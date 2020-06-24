# Mest Example
Using mest for serving models


## Environment setup
Recommended to run with virtualenv/pipenv/poetry

Python version needed: 3.6+
##
This demo requires docker-compose for full functionality

`pip install -r requirements.txt`

`docker-compose up`


## High Level Flow

This project contains 2 parts: model training and model serving.

### Model Training
executing `python main_train.py` - will train the model and store it locally and in minio

### Model Serving
executing `./run_service.sh` - will do the following:
* run `python download_models.py` - which takes the latest model stores on minio and downloads it to a local folder
* run `python main_mest.py` will load the local model files and serve an endpoint for inference
