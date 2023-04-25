import logging

import cv2
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import torch

from spoof.utils.transforms import FaceRegionRCXT, MetaAddLMSquare

logger = logging.getLogger("spoofds")
logger.setLevel(logging.INFO)


class CASIA(Dataset):
    def __init__(
        self,
        annotations_file: str,
    ):
        self.annotations = pd.read_csv(annotations_file)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = self.annotations.iloc[idx, 0]
        img_cv2 = cv2.imread(img_path)
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

        label = self.annotations.iloc[idx, -1]

        # Transforms
        face_rect = self.annotations.iloc[idx, 1:5].values.astype(int)
        face_landmark = (
            self.annotations.iloc[idx, 5:-1].values.astype(int).reshape(-1, 2)
        )
        meta = {"face_rect": face_rect, "face_landmark": face_landmark}
        sample = {"image": img_cv2, "meta": meta}

        # Transform and reshape to (C, H, W)
        transformed_sample = self._transform(sample)
        transformed_img = self._transform(sample)["image"].transpose((2, 0, 1))

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
