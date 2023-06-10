import logging

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from spoof.dataset.transforms import FaceRegionRCXT, MetaAddLMSquare
from spoof.dataset.transforms_img import (
    ColorJitterCV,
    RandomGaussianBlur,
    RandomHorizontalFlip,
)

logger = logging.getLogger("spoofds")
logger.setLevel(logging.INFO)


def align(sample):
    return Compose(
        [
            FaceRegionRCXT(size=(224, 224)),
            MetaAddLMSquare(),
        ]
    )(sample)


def augment(sample):
    # Apply ColorJitterCV
    color_jitter = ColorJitterCV(
        brightness=0.8, contrast=0.1, gamma=0.2, temp=0.8, p=0.75
    )
    sample = color_jitter(sample)

    # Apply RandomGaussianBlur
    blur = RandomGaussianBlur()
    sample = blur(sample)

    # Apply RandomHorizontalFlip
    flip = RandomHorizontalFlip()
    sample = flip(sample)

    return sample


class FaceDataset(Dataset):
    def __init__(self, annotations_file: str, mode: str = "train"):
        super(FaceDataset, self).__init__()
        self.annotations = pd.read_csv(annotations_file)
        self.mode = mode

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self._load_sample(idx)
        sample = align(sample)

        if self.mode == "train":
            sample = augment(sample)

        sample = self._normalize(sample)

        return sample

    def _load_sample(self, idx):
        row = self.annotations.iloc[idx]
        img_path = row["image_file"]

        face_rect = row["face_rect_x":"face_rect_height"].values
        face_landmark = row["landmark_1":"landmark_14"].values.reshape((-1, 2))

        sample = {
            "image": cv2.imread(img_path),
            "meta": {"face_rect": face_rect, "face_landmark": face_landmark},
            "label": row["label"],
            "filename": img_path,
        }

        return sample

    def _normalize(self, sample):
        img = sample["image"]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)  # Convert to (C, H, W) format
        img = torch.from_numpy(img).float() / 255.0  # Convert to torch tensor
        img = torch.clamp(img, 0.0, 1.0)  # Clamp values to [0, 1]

        sample["image"] = img

        return sample


class FaceDatasetLeaveOneOut(FaceDataset):
    def __init__(
        self,
        annotations_file: str,
        mode: str = "train",
        spoof_type: str = None,
    ):
        super(FaceDatasetLeaveOneOut, self).__init__(annotations_file, mode)
        self.spoof_type = spoof_type

        if spoof_type is not None:
            self._leave_out_spoof_type()

    def _leave_out_spoof_type(self):
        if self.spoof_type is not None:
            self.annotations = self.annotations[
                self.annotations["spoof_type"] != self.spoof_type
            ]

    def leave_out_all_except(self, spoof_type):
        self.annotations = self.annotations[
            (self.annotations["spoof_type"] == spoof_type)
            | (self.annotations["spoof_type"] == "live")
        ]
        self.spoof_type = spoof_type

    def __repr__(self) -> str:
        if self.spoof_type is None:
            spoof_type_str = "all"
        else:
            spoof_type_str = self.spoof_type
        return (
            f"FaceDatasetLeaveOneOut(mode={self.mode}, len={len(self)}, "
            f"spoof={spoof_type_str})"
        )
