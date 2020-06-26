## Download models from minio
python download_models.py

## Run the inference service
python main_mest.py

## You can also run the service on gunicorn:
# gunicorn -b 0.0.0.0:5000 -k gevent -w 4 main_mest:app