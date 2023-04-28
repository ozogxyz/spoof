import os
import hydra
from omegaconf import OmegaConf
import torch
import yaml


def test_casia_train(config_training):
    casia_train_ds = hydra.utils.instantiate(
        config_training["data"]["train"], _recursive_=False
    )

    img = casia_train_ds[0]["image"]
    assert img.shape == (3, 224, 224)
    assert img.dtype == torch.float32
    assert img.max() <= 1
    assert img.min() >= 0

    label = casia_train_ds[0]["label"]
    assert label == 0 or label == 1

    filename = casia_train_ds[0]["filename"]
    assert os.path.exists(filename)

    assert len(casia_train_ds) == 44900


def test_casia_val(config_val):
    casia_val = hydra.utils.instantiate(config_val["val_base"])

    img = casia_val[0]["image"]
    assert img.shape == (3, 224, 224)
    assert img.dtype == torch.float32
    assert img.max() <= 1
    assert img.min() >= 0

    label = casia_val[0]["label"]
    assert label == 0 or label == 1

    filename = casia_val[0]["filename"]
    assert os.path.exists(filename)

    assert len(casia_val) == 10618


def test_casia_test(config_test):
    casia_test = hydra.utils.instantiate(config_test["val_base"])

    img = casia_test[0]["image"]
    assert img.shape == (3, 224, 224)
    assert img.dtype == torch.float32
    assert img.max() <= 1
    assert img.min() >= 0

    label = casia_test[0]["label"]
    assert label == 0 or label == 1

    filename = casia_test[0]["filename"]
    assert os.path.exists(filename)

    assert len(casia_test) == 54906
