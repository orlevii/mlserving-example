from config import config
from iris_classifier import ModelTrainer
from utils.minio import minio_client
from utils.model_version import ModelVersion


def create_bucket_if_needed():
    # Used for first run...
    if not minio_client.client.bucket_exists(config.ML_BUCKET):
        minio_client.client.make_bucket(config.ML_BUCKET)


def main():
    create_bucket_if_needed()
    trainer = ModelTrainer()
    trainer.run()

    model_version = ModelVersion.find_or_create(config.MODEL_NAME)
    print(f'Uploading model: {model_version.model_name}')
    model_version.bump(config.LOCAL_MODEL_DIR)

    print(model_version.remote_directory)
    print('DONE!')


if __name__ == '__main__':
    main()
