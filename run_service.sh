## Download latest models from minio
# python download_models.py

## Run the inference service
python mlserving_main.py

## You can also run the service on gunicorn:
## RUN: pip install gunicorn[gevent]
# gunicorn -b 0.0.0.0:5000 -k gevent -w 4 mlserving_main:app
