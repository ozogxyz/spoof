import cv2
import numpy as np
import pytest
import torch

from src.spoof.dataset.dataset import FaceDataset
from src.spoof.utils.visualize import draw_face_rectangle, draw_landmarks


@pytest.fixture
def ds():
    return FaceDataset("tests/data/annotations.csv")


def test_visualize(ds):
    idx = np.random.randint(0, len(ds))
    img = ds.annotations.iloc[idx, 0]
    img = cv2.imread(img)[:, :, ::-1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # show_frame(img)
    cv2.imwrite("tests/data/plain.jpg", img)

    face_rect = ds.annotations.iloc[idx, 1:5].values.astype(np.int32)
    face_landmark = ds.annotations.iloc[idx, 5:-1].values.reshape((-1, 2))
    img = draw_face_rectangle(img, face_rect)
    img = draw_landmarks(img, face_landmark)
    # show_frame(img)
    cv2.imwrite("tests/data/rect_landmark.jpg", img)

    meta = {"face_rect": face_rect, "face_landmark": face_landmark}
    sample = {"image": img, "meta": meta}
    transformed_img = ds._transform(sample)["image"].transpose((2, 0, 1))
    transformed_img = torch.tensor(transformed_img, dtype=torch.float32) / 255
    transformed_img = torch.clamp(transformed_img, 0, 1)
    transformed_img = transformed_img.numpy().transpose((1, 2, 0))
    transformed_img = (transformed_img * 255).astype(np.uint8)
    # show_frame(transformed_img)
    cv2.imwrite("tests/data/transformed.jpg", transformed_img)
