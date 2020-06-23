from iris_classifier import ModelTrainer


def main():
    trainer = ModelTrainer()
    trainer.run()
    # TODO: upload to storage
    # TODO: simple versioning


if __name__ == '__main__':
    main()
