from minio import Minio

from config import config

minioClient = Minio(f'{config.MINIO_HOST}:{config.MINIO_PORT}',
                    access_key=config.MINIO_ACCESS_KEY,
                    secret_key=config.MINIO_SECRET_KEY,
                    secure=False)

for obj in minioClient.list_objects(config.ML_BUCKET):
    print(obj.object_name)
