import logging

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

import sys

sys.path.append("../spoof")
import spoof
from spoof.utils.transforms import FaceRegionRCXT, MetaAddLMSquare
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor, Normalize

logger = logging.getLogger("spoofds")
logger.setLevel(logging.INFO)


casia_transform = Compose(
    [
        FaceRegionRCXT(crop=(224, 224)),
        MetaAddLMSquare(),
    ]
)


class CasiaDataset(Dataset):
    def __init__(
        self,
        annotations_file: str,
        num_samples: int,
        image_size: int,
        transforms=casia_transform,
    ):
        self.annotations = pd.read_csv(annotations_file)
        self.num_samples = num_samples
        self.image_size = image_size
        self.transforms = transforms

    def __len__(self):
        return len(self.annotations)

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
        img_transformed = self.transforms(sample)

        norm = Compose([ToTensor()])
        img_tensor = norm(img_transformed["image"])

        filename = f"sample_num_{idx:04d}.png"

        sample_dict = {
            "image": img_tensor,
            "label": label,
            "filename": filename,
        }
        return sample_dict


if __name__ == "__main__":
    dataset = CasiaDataset("data/casia/images/train/train.csv", 45000, 224)
    print(dataset[0])
