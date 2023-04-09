import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from models.cnn import CNN
from preprocessing.casia import CASIA
from preprocessing.transforms import FaceRegionRCXT


@hydra.main(version_base=None, config_path="../", config_name="config")
def main(cfg: DictConfig):
    # Extract frames and metadata
    # extract_frames(cfg.data.train_video, cfg.data.train_frames)
    # extract_metadata(cfg.data.train_meta, cfg.data.train_frames)
    # Load data
    train_ds = CASIA(
        cfg.data.train,
        cfg.data.annotations,
        cfg.data.train_meta,
        transform=FaceRegionRCXT(size=cfg.data.face_region.size),
    )
    train_dl = DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True)

    # Load model
    model = CNN(**cfg.model)

    # Train model
    for epoch in range(cfg.train.epochs):
        for batch in train_dl:
            image, label = batch["image"], batch["label"]
            print(image.shape, label.shape)
            break


if __name__ == "__main__":
    main()
