import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import csv
from src.preprocessing.create_sample import CreateSample

sampler = CreateSample()


class CASIA(Dataset):
    """CASIA dataset.

    Args:
        annotations: path to the csv file with labels
        transform: transform to apply to the sample
    """

    def __init__(self, annotations_file: str, transform) -> None:
        # read the labels from the csv file
        self.annotations = self._get_annotations(annotations_file)
        self.transform = transform

    def _get_annotations(self, annotations_file: str):
        with open(annotations_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            next(csv_reader, None)
            annotations = []
            for row in csv_reader:
                annotations.append(row)

        return annotations

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int):
        frame_path, metadata_path, label = self.annotations[idx]

        frame = cv2.imread(frame_path)
        metadata = json.load(open(metadata_path))
        label = torch.tensor(int(label), dtype=torch.int)

        sample = sampler.create_sample(metadata, frame)

        if self.transform:
            sample = self.transform(sample)
            frame = sample["image"]

        # convert open cv image to pytorch image
        tensor_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(
            np.transpose(tensor_frame, (2, 0, 1)).astype(np.float32) / 255.0
        )
        return frame, label
