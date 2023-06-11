import hydra
import torch


def test_ds_train(config_training):
    conf_train_sys = config_training["training_system"]
    train_sys = hydra.utils.instantiate(conf_train_sys, _recursive_=False)
    train_ds = train_sys.train_dataloader().dataset

    sample = train_ds[0]
    assert sample["image"].shape == (3, 224, 224)
    assert sample["image"].dtype == torch.float32
    assert sample["image"].max() <= 1
    assert sample["image"].min() >= 0

    assert sample["label"] == 0 or sample["label"] == 1

    x = sample["image"]
    log_str = f"Tensor range: [{x.min().cpu().item():.4f}, {x.max().cpu().item():.4f}]"
    print(log_str, flush=True)

    assert False
