# test instantiations
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from hydra.utils import instantiate


def test_config_training():
    with open("config/train_lora.yaml", "r") as f:
        config_training = yaml.load(f, Loader=yaml.FullLoader)

    config_system = config_training["training_system"]

    training_system = instantiate(config_system, _recursive_=False)
    assert isinstance(training_system, pl.LightningModule)

    model = instantiate(config_system["model"], _recursive_=False)
    assert model.__class__.__name__ == config_system["model"]["_target_"].split(".")[-1]
    assert isinstance(model, nn.Module)

    ds_train = instantiate(config_system["data"]["train"], _recursive_=False)
    assert isinstance(ds_train, torch.utils.data.Dataset)
    assert len(ds_train) > 0

    ds_val = instantiate(config_system["data"]["val_base"], _recursive_=False)
    assert isinstance(ds_val, torch.utils.data.Dataset)
    assert len(ds_val) > 0
