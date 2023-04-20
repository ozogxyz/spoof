import logging
import sys

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.append("../spoof")
from torchvision.transforms import Compose, Normalize, ToTensor

import spoof
from spoof.utils.transforms import FaceRegionRCXT, MetaAddLMSquare

logger = logging.getLogger("spoofds")
logger.setLevel(logging.INFO)


class CasiaDataset(Dataset):
    def __init__(
        self,
        annotations_file: str,
        num_samples: int,
        image_size: int,
    ):
        self.annotations = pd.read_csv(annotations_file)
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img_path = self.annotations.iloc[idx, 0]
        img_cv2 = cv2.imread(img_path)
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        # img_tensor = torch.from_numpy(img_cv2).permute(2, 0, 1).float()

        label = self.annotations.iloc[idx, -1]

        # Transforms
        face_rect = self.annotations.iloc[idx, 1:5].values.astype(int)
        face_landmark = self.annotations.iloc[idx, 5:-1].values.astype(int).reshape(-1, 2)
        inputs = {
            "image": img_cv2,
            "face_rect": face_rect,
            "face_landmark": face_landmark,
            "label": label,
        }
        meta = {"face_rect": face_rect, "face_landmark": face_landmark}
        sample = {"image": img_cv2, "meta": meta}
        img_transformed = self._transforms(sample)

        norm = Compose([ToTensor()])
        img_tensor = norm(img_transformed["image"])

        filename = f"sample_num_{idx:04d}.png"

        sample_dict = {
            "image": img_tensor,
            "label": label,
            "filename": filename,
        }

        return sample_dict

    def _transforms(self, sample):
        tr = Compose(
            [
                FaceRegionRCXT(crop=(self.image_size, self.image_size)),
                MetaAddLMSquare(),
            ]
        )
        return tr(sample)


if __name__ == "__main__":
    dataset = CasiaDataset("data/casia/images/train/train.csv", 45000, 224)
    print(dataset[0])
