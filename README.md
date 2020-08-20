# mlserving example
Using mlserving for serving models

For this example, we will use `mlserving` as the model-server.

`mlserving` requires a model, so I've also implemented a small training part for that. 

## Environment setup
Recommended to run with virtualenv

Python version needed: 3.6+
##
This demo requires docker-compose for full functionality

`pip install -r requirements.txt`

`docker-compose up`

## High Level Flow

This project contains 2 parts: model training and model serving (using `mlserving`).

### Model Training
executing `python train_main.py` - will train the model and store it locally and in minio

### Model Serving
executing `./run_service.sh` - will do the following:
* run `python download_models.py` - which takes the latest model stores on minio and downloads it to a local folder
    * This is optional, demonstrating a simple (and naive) flow of downloading the latest trained model.
* run `python mlserving_main.py` will load the local model files and serve an endpoint for inference
