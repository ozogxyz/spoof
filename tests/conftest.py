import pytest
import yaml


@pytest.fixture(scope="module")
def config_training():
    with open("config/train.yaml", "r") as f:
        config_training = yaml.load(f, Loader=yaml.FullLoader)
        return config_training


@pytest.fixture(scope="module")
def config_test():
    with open("config/test.yaml", "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)
