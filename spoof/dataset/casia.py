import logging
import sys

import cv2
import pandas as pd
from torch.utils.data import Dataset

sys.path.append("../spoof")
from torchvision.transforms import Compose

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
        # img_tensor = torch.from_numpy(img_cv2).permute(2, 0, 1).float()

        label = self.annotations.iloc[idx, -1]

        # Transforms
        face_rect = self.annotations.iloc[idx, 1:5].values.astype(int)
        face_landmark = self.annotations.iloc[idx, 5:-1].values.astype(int).reshape(-1, 2)
        meta = {"face_rect": face_rect, "face_landmark": face_landmark}
        sample = {"image": img_cv2, "meta": meta}
        img_transformed = self._transforms(sample)

        filename = f"{img_path}.jpg"

        sample_dict = {
            "image": img_transformed,
            "label": label,
            "filename": filename,
        }

        return sample_dict

    def _transforms(self, sample):
        tr = Compose(
            [
                MetaAddLMSquare(),
                FaceRegionRCXT(size=(224, 224)),
            ]
        )
        return tr(sample)
