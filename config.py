import os
import sys

from dotenv import load_dotenv

load_dotenv()

# minio access from non-linux
if 'linux' not in sys.platform.lower():
    os.environ['MINIO_HOST'] = '127.0.0.1'


class Config:
    MINIO_HOST = os.getenv('MINIO_HOST')
    MINIO_PORT = os.getenv('MINIO_PORT')
    MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
    MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY')
    ML_BUCKET = 'ml-models'
    LOCAL_MODEL_DIR = 'model_artifacts'
    MODEL_NAME = 'iris_classifier'


config = Config()
