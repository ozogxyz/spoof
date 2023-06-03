import logging
import os

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from spoof.dataset.transforms import FaceRegionRCXT, MetaAddLMSquare

logger = logging.getLogger("spoofds")
logger.setLevel(logging.INFO)


class FaceDataset(Dataset):
    def __init__(
        self,
        annotations_file: str,
    ):
        self.annotations = pd.read_csv(annotations_file)
        self.spoof_type = None

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = self.annotations.iloc[idx, 0]
        if not os.path.exists(img_path):
            logger.debug(f"Image {img_path} not found")
            raise FileNotFoundError(f"Image {img_path} not found")
        img_cv2 = cv2.imread(img_path)
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

        # Get face rect and landmark
        face_rect = self.annotations.iloc[idx, 1:5].values
        face_landmark = self.annotations.iloc[idx, 5:-2].values.reshape(
            (-1, 2)
        )
        meta = {"face_rect": face_rect, "face_landmark": face_landmark}
        sample = {"image": img_cv2, "meta": meta}

        # Transform and reshape to (C, H, W), to tensor
        transformed_img = self._transform(sample)["image"].transpose((2, 0, 1))
        transformed_img = (
            torch.tensor(transformed_img, dtype=torch.float32) / 255
        )

        # Clamp img to [0, 1]
        transformed_img = torch.clamp(transformed_img, 0, 1)

        # annotations may not have labels in last column for all datasets
        label = self.annotations.iloc[
            idx, self.annotations.columns.get_loc("class_label")
        ]

        sample_dict = {
            "image": transformed_img,
            "label": label,
            "filename": img_path,
        }

        return sample_dict

    def _transform(self, sample):
        tr = Compose(
            [
                MetaAddLMSquare(),
                FaceRegionRCXT(size=(224, 224)),
            ]
        )
        return tr(sample)

    def __repr__(self) -> str:
        return f"FaceDataset({len(self.annotations)} samples)"

    def leave_out(self, spoof_type: str):
        self.annotations = self.annotations[
            self.annotations["spoof_type"] != spoof_type
        ]
        self.spoof_type = spoof_type

    def leave_out_all_except(self, spoof_type):
        self.annotations = self.annotations[
            (self.annotations["spoof_type"] == spoof_type)
            | (self.annotations["spoof_type"] == "live")
        ]
        self.spoof_type = spoof_type
