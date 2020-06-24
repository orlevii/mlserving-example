from config import config
from utils.model_version import ModelVersion


def main():
    model_version = ModelVersion.find_or_create(config.MODEL_NAME)
    print(f'Downloading model: {model_version.model_name}, version: {model_version.version}')
    print(model_version.remote_directory)
    model_version.get(config.LOCAL_MODEL_DIR)


if __name__ == '__main__':
    main()
