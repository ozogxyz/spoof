import hydra
import timm
import torchsummary
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from datasets import CASIA
import torch


@hydra.main(version_base=None, config_path="..", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg.model))

    model = timm.create_model(cfg.model.name, **cfg.model.params)

    torchsummary.summary(model, (3, 224, 224), device="cuda")

    # Casia dataset and dataloader
    casia = CASIA(**cfg.dataset.params)

    # Split the dataset into train and validation
    train_size = int(0.8 * len(casia))
    val_size = len(casia) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        casia, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    # Train the model
    for epoch in range(cfg.train.epochs):
        for i, (samples, labels) in enumerate(train_loader):
            images = samples["image"]
            print(
                f"Epoch: {epoch}, Batch: {i}, Images: {images.shape}, Labels: {labels.shape}"
            )

        for i, (images, labels) in enumerate(val_loader):
            print(
                f"Epoch: {epoch}, Batch: {i}, Images: {images.shape}, Labels: {labels.shape}"
            )


if __name__ == "__main__":
    main()
