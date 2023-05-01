import pytest
import yaml


@pytest.fixture(scope="module")
def config_training():
    with open("config/train.yaml", "r") as f:
        config_training = yaml.load(f, Loader=yaml.FullLoader)
        return config_training["training_system"]


# @pytest.fixture(scope="module")
# def config_val():
#     with open("config/validate.yaml", "r") as f:
#         return yaml.load(f, Loader=yaml.FullLoader)


# @pytest.fixture(scope="module")
# def config_test():
#     with open("config/test.yaml", "r") as f:
#         return yaml.load(f, Loader=yaml.FullLoader)
