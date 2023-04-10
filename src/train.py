import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from models.cnn import CNN
from preprocessing.casia import CASIA
from preprocessing.transforms import FaceRegionRCXT
from preprocessing.casia import extract_frames, extract_metadata
from utils.annotations import create_annotations


@hydra.main(version_base=None, config_path="../", config_name="config")
def main(cfg: DictConfig):
    # Extract frames and metadata
    if cfg.data.extract:
        extract_frames(cfg.data.train_videos, cfg.data.train_frames)
        extract_metadata(cfg.data.train_meta, cfg.data.train_frames)
        create_annotations(cfg.data.train_frames, cfg.data.annotations)
        create_annotations(cfg.data.train_frames, cfg.data.train_meta)

    # Load data
    train_ds = CASIA(
        cfg.data.train_videos,
        cfg.data.annotations,
        cfg.data.train_meta,
        transform=FaceRegionRCXT(size=cfg.data.face_region.size),
    )
    train_dl = DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=False)

    # Load model
    model = CNN(**cfg.model)

    # Train model
    count = 0
    for epoch in range(cfg.train.epochs):
        for batch in train_dl:
            image, label = batch
            count += 1
            print(image.shape, label.shape)
        print(f"Number of epochs: {epoch}")

    print(f"Number of batches: {count}")


if __name__ == "__main__":
    main()
