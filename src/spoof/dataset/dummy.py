import logging


import torch
from torch.utils.data import Dataset


logger = logging.getLogger("spoofds")
logger.setLevel(logging.INFO)


class DummyDataset(Dataset):
    def __init__(self, num_samples=512, image_size=128):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        label = torch.randint(0, 2, (1,))

        img_tensor = torch.randn(3, self.image_size, self.image_size)
        img_tensor = img_tensor.clamp(0.0, 1.0)

        filename = f"sample_num_{idx:04d}.png"

        sample_dict = {
            "image": img_tensor,
            "label": label,
            "filename": filename,
        }
        return sample_dict
