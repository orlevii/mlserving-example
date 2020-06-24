import os

from config import config
from .minio import minio_client


class ModelVersion:
    """
    Naive model versioning implementation
    """

    def __init__(self, model_name, version):
        self.model_name = model_name
        self.version = version

    def bump(self, artifacts_path):
        """
        :param artifacts_path: Local artifacts path to upload
        :return:
        """
        new_ver = self.version + 1
        dst_directory = os.path.join(self.model_name, 'versions', str(new_ver))
        minio_client.upload_directory(artifacts_path, config.ML_BUCKET, dst_directory)

        minio_client.write_object(config.ML_BUCKET, self.version_path(self.model_name), str(new_ver))
        self.version = new_ver

    def get(self, artifacts_path):
        """
        :param artifacts_path: Local artifacts path to download to
        :return:
        """
        to_download = self.remote_directory
        minio_client.download_directory(config.ML_BUCKET, to_download, artifacts_path)

    @property
    def remote_directory(self):
        return os.path.join(self.model_name, 'versions', str(self.version))

    @classmethod
    def find_or_create(cls, model_name):
        version_path = cls.version_path(model_name)
        if minio_client.object_exists(config.ML_BUCKET, version_path):
            version = minio_client.read_object(config.ML_BUCKET, version_path).decode('utf-8')
        else:
            version = '0'
            minio_client.write_object(config.ML_BUCKET, version_path, version)

        return ModelVersion(model_name, int(version))

    @classmethod
    def version_path(cls, model_name):
        return os.path.join(model_name, 'current_version')
