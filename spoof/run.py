import logging

import hydra
import timm
import torch
from datapipe import build_datapipes
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from trainer import trainer


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="..", config_name="config")
def main(cfg: DictConfig) -> None:
    # Build train and test datapipes and dataloaders
    logger.info("Building train and test datapipes")
    train_ds = build_datapipes(cfg.data.train)
    test_ds = build_datapipes(cfg.data.test)
    assert train_ds is not None, logger.error("Error creating train datapipe")
    assert test_ds is not None, logger.error("Error creating test datapipe")
    logger.info("Train and test datapipes created")

    logger.info("Building train and test dataloaders")
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=cfg.data.batch_size)
    test_dl = DataLoader(test_ds, shuffle=False, batch_size=cfg.data.batch_size)
    assert train_dl is not None, logger.error("Error creating train dataloader")
    assert test_dl is not None, logger.error("Error creating test dataloader")
    logger.info("Train and test dataloaders created")

    # Model, loss and optimizer
    logger.info("Creating model, loss and optimizer")
    model = timm.create_model(cfg.model.name, pretrained=True).to(cfg.device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    assert model is not None, logger.error("Error creating model")
    assert criterion is not None, logger.error("Error creating loss")
    assert optimizer is not None, logger.error("Error creating optimizer")
    logger.info("Model, loss and optimizer created")

    # Get the pretrained weights
    pretrained_weights = model.state_dict()
    assert pretrained_weights is not None, logger.error("Error getting weights")
    logging.info("Pretrained weights loaded")

    # Train the model
    logger.info("Training model")
    trainer(cfg, model, train_dl, criterion, optimizer)
    logger.info("Model trained")

    # Save the model
    logger.info("Saving model")
    torch.save(model.state_dict(), cfg.model.save_path)
    logger.info("Model saved to {}".format("model.ckpt"))


if __name__ == "__main__":
    main()
