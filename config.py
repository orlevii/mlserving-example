import os
import sys

from dotenv import load_dotenv

load_dotenv()

# minio access from non-linux
if 'linux' not in sys.platform.lower():
    os.environ['MINIO_HOST'] = '127.0.0.1:{port}'.format(
        port=os.environ['MINIO_HOST'].split(':')[1]
    )


class Config:
    MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT')
    MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
    MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY')
    MINIO_SECURE = os.getenv('SECURE', 'false') == 'true'
    ML_BUCKET = os.getenv('MINIO_BUCKET')
    LOCAL_MODEL_DIR = 'model_artifacts'
    MODEL_NAME = 'iris_classifier'


config = Config()
