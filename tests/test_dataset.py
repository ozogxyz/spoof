import os

import hydra
import torch


def test_ds_train(config_training):
    train_ds = hydra.utils.instantiate(
        config_training["data"]["train"], _recursive_=False
    )

    img = train_ds[0]["image"]
    assert img.shape == (3, 224, 224)
    assert img.dtype == torch.float32
    assert img.max() <= 1
    assert img.min() >= 0

    label = train_ds[0]["label"]
    assert label == 0 or label == 1

    filename = train_ds[0]["filename"]
    assert os.path.exists(filename)


def test_ds_val(config_training):
    val_ds = hydra.utils.instantiate(
        config_training["data"]["val_base"], _recursive_=False
    )

    img = val_ds[0]["image"]
    assert img.shape == (3, 224, 224)
    assert img.dtype == torch.float32
    assert img.max() <= 1
    assert img.min() >= 0

    label = val_ds[0]["label"]
    assert label == 0 or label == 1

    filename = val_ds[0]["filename"]
    assert os.path.exists(filename)
