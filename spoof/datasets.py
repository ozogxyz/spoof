import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CASIA(Dataset):
    """CASIA dataset."""

    def __init__(self, annotations_path: str, root_dir: str, transform=None):
        """
        Args:
            annotations_path (string): Path to the annotations file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(annotations_path, "r") as f:
            self.annotations = json.load(f)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir, self.annotations[idx][0])

        face_rect = self.annotations[idx][1]
        face_landmarks = self.annotations[idx][2]
        face_landmarks = np.array(face_landmarks).reshape(-1, 2)
        label = self.annotations[idx][3]

        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        meta = {"face_rect": face_rect, "face_landmark": face_landmarks}
        sample = {"image": image, "meta": meta}

        if self.transform:
            sample = self.transform(sample)

        return sample, label
