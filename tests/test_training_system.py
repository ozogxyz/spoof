import hydra
import pytest
import torch


@pytest.mark.slow
def test_training_system(config_training):
    training_system = hydra.utils.instantiate(
        config_training, _recursive_=False
    )

    train_dl = training_system.train_dataloader()
    assert train_dl

    batch_dict = next(iter(train_dl))
    img = batch_dict["image"]

    # Check the model forward pass
    model = training_system.model
    assert model is not None
    assert isinstance(model, torch.nn.Module)

    out_dict = model(img)
    out_logit = out_dict["out_logit"]
    assert out_logit.shape == (config_training["train_batch_size"], 1)
    assert out_logit.dtype == torch.float32

    score = model.get_liveness_score(out_dict)
    assert score.shape == (config_training["train_batch_size"], 1)
    assert score.dtype == torch.float32
    assert score.max() <= 1.0
    assert score.min() >= 0.0

    # Check the loss function
    loss_func = training_system.loss_func
    assert loss_func is not None
    assert isinstance(loss_func, torch.nn.Module)
    allvars = dict(batch_dict, **out_dict)
    loss, details = loss_func(**allvars)
    assert loss.item() >= 0.0
    assert loss.dtype == torch.float32
    assert isinstance(details, dict)


def test_training_system_fast(config_training):
    training_system = hydra.utils.instantiate(
        config_training, _recursive_=False
    )

    batch_dict = {
        "image": torch.rand(2, 3, 224, 224),
        "label": torch.randint(0, 2, (2,)),
        "filename": "test.jpg",
    }

    img = batch_dict["image"]

    # Check the model forward pass
    model = training_system.model
    assert model is not None
    assert isinstance(model, torch.nn.Module)

    out_dict = model(img)
    out_logit = out_dict["out_logit"]
    assert out_logit.shape == (2, 1)
    assert out_logit.dtype == torch.float32

    score = model.get_liveness_score(out_dict)
    assert score.shape == (2, 1)
    assert score.dtype == torch.float32
    assert score.max() <= 1.0
    assert score.min() >= 0.0

    # Check the loss function
    loss_func = training_system.loss_func
    assert loss_func is not None
    assert isinstance(loss_func, torch.nn.Module)
    allvars = dict(batch_dict, **out_dict)
    loss, details = loss_func(**allvars)
    assert loss.item() >= 0.0
    assert loss.dtype == torch.float32
    assert isinstance(details, dict)
