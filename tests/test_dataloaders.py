import hydra


def test_train_dataloader(config_training):
    training_system = hydra.utils.instantiate(
        config_training, _recursive_=False
    )

    train_dl = training_system.train_dataloader()

    assert train_dl is not None

    img_train = next(iter(train_dl))["image"]

    # Check that the dataloader return the correct shapes, BxCxHxW
    assert img_train.shape[1:] == (3, 224, 224)

    # Check that the dataloader return normalized images
    assert img_train.flatten().max() <= 1.0
    assert img_train.flatten().min() >= 0.0
